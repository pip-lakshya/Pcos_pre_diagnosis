"""Free OpenStreetMap doctor search with respectful public-service caching."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from app.config import settings

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_FALLBACK_URLS = (
    "https://overpass.kumi.systems/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)
CACHE_TTL_SECONDS = 24 * 60 * 60
GEOCODE_TTL_SECONDS = 24 * 60 * 60
MIN_LOCAL_RESULTS = 3
WOMENS_HEALTH = re.compile(
    r"gynae|gynec|obstet|women|woman|female|maternity|fertility",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Coordinates:
    lat: float
    long: float


class OpenStreetMapDoctorFinder:
    def __init__(self) -> None:
        self._geocode_cache: dict[str, tuple[float, Coordinates]] = {}
        self._overpass_cache: dict[tuple[float, float, int], tuple[float, list[dict]]] = {}
        self._geocode_lock = asyncio.Lock()
        self._overpass_lock = asyncio.Lock()
        self._last_geocode_request = 0.0

    @property
    def user_agent(self) -> str:
        contact = settings.osm_contact_email.strip()
        suffix = f" (contact: {contact})" if contact else ""
        return f"Narisaarthi-PCOS-App/1.0{suffix}"

    async def _fetch_geocode(self, query: str) -> Coordinates | None:
        params = {"q": query, "format": "json", "limit": 1}
        # Nominatim searches globally by default. Six digit Indian postcodes are
        # much more reliable when scoped to India, which is the app's audience.
        if re.fullmatch(r"\d{6}", query.strip()):
            params["countrycodes"] = "in"
        url = f"{NOMINATIM_URL}?{urlencode(params)}"
        results = await asyncio.to_thread(
            self._request_json,
            Request(url, headers={"User-Agent": self.user_agent, "Accept": "application/json"}),
            10,
        )
        if not results:
            return None
        return Coordinates(lat=float(results[0]["lat"]), long=float(results[0]["lon"]))

    async def geocode(self, query: str) -> Coordinates | None:
        key = " ".join(query.casefold().split())
        cached = self._geocode_cache.get(key)
        now = time.monotonic()
        if cached and cached[0] > now:
            return cached[1]
        # Serialize Nominatim access and enforce at most one uncached request per second.
        async with self._geocode_lock:
            cached = self._geocode_cache.get(key)
            now = time.monotonic()
            if cached and cached[0] > now:
                return cached[1]
            delay = 1.0 - (now - self._last_geocode_request)
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_geocode_request = time.monotonic()
            coordinates = await self._fetch_geocode(query)
            if coordinates is not None:
                self._geocode_cache[key] = (time.monotonic() + GEOCODE_TTL_SECONDS, coordinates)
            return coordinates

    @staticmethod
    def _overpass_query(lat: float, long: float, radius_m: int) -> str:
        return f"""[out:json][timeout:18];
(
 nwr[\"healthcare\"~\"^(gynaecologist|gynecologist|obstetrician)$\",i](around:{radius_m},{lat},{long});
  nwr[\"healthcare:speciality\"~\"gynae|gynec|obstet|women|maternity|fertility\",i](around:{radius_m},{lat},{long});
  nwr[\"healthcare:specialty\"~\"gynae|gynec|obstet|women|maternity|fertility\",i](around:{radius_m},{lat},{long});
  nwr[\"amenity\"~\"^(doctors|clinic)$\"][\"healthcare:speciality\"~\"gynae|gynec|obstet|women|maternity|fertility\",i](around:{radius_m},{lat},{long});
  nwr[\"amenity\"~\"^(doctors|clinic)$\"][\"healthcare:specialty\"~\"gynae|gynec|obstet|women|maternity|fertility\",i](around:{radius_m},{lat},{long});
  nwr[\"amenity\"=\"clinic\"][\"name\"~\"gynae|gynec|obstet|women|woman|female|maternity|fertility\",i](around:{radius_m},{lat},{long});
);
out center tags;"""

    @staticmethod
    def _is_womens_health(tags: dict) -> bool:
        category = " ".join(str(tags.get(key, "")) for key in (
            "healthcare", "healthcare:speciality", "healthcare:specialty", "medical_specialty", "speciality", "specialty", "name"
        ))
        return bool(WOMENS_HEALTH.search(category))

    @staticmethod
    def _address(tags: dict) -> str:
        full = tags.get("addr:full")
        if full:
            return str(full)
        parts = [tags.get("addr:housenumber"), tags.get("addr:street"), tags.get("addr:suburb"), tags.get("addr:city"), tags.get("addr:postcode")]
        assembled = ", ".join(str(part).strip() for part in parts if part)
        return assembled or "Address not listed"

    @staticmethod
    def _request_json(request: Request, timeout: int) -> dict | list:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    @classmethod
    def _parse_elements(cls, response: dict) -> list[dict]:
        results = []
        seen: set[tuple[str, str]] = set()
        for element in response.get("elements", []):
            tags = element.get("tags") or {}
            if not cls._is_womens_health(tags):
                continue
            coordinates = element if "lat" in element and "lon" in element else element.get("center") or {}
            if "lat" not in coordinates or "lon" not in coordinates:
                continue
            lat, long = float(coordinates["lat"]), float(coordinates["lon"])
            identity = (str(element.get("type", "")), str(element.get("id", "")))
            if identity in seen:
                continue
            seen.add(identity)
            results.append({
                "name": tags.get("name") or "Unnamed healthcare provider",
                "address": cls._address(tags),
                "phone": tags.get("contact:phone") or tags.get("phone") or "not listed",
                "lat": lat,
                "long": long,
                "directions_url": f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={long:.6f}",
            })
        return results

    async def _fetch_overpass_from(self, endpoint: str, lat: float, long: float, radius_m: int) -> list[dict]:
        query = self._overpass_query(lat, long, radius_m)
        body = urlencode({"data": query}).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={"User-Agent": self.user_agent, "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        data = await asyncio.to_thread(self._request_json, request, 18)
        return self._parse_elements(data)

    async def _fetch_overpass(self, lat: float, long: float, radius_m: int) -> list[dict]:
        # Public Overpass instances occasionally rate-limit or time out. Retry the
        # same cached query against two documented public mirrors before failing.
        last_error: Exception | None = None
        for endpoint in (OVERPASS_URL, *OVERPASS_FALLBACK_URLS):
            try:
                return await self._fetch_overpass_from(endpoint, lat, long, radius_m)
            except Exception as error:
                last_error = error
                # Do not log the searched city, postcode, or coordinates.
                logger.warning("Doctor search endpoint failed (%s)", type(error).__name__)
        if last_error is not None:
            raise RuntimeError("All public OpenStreetMap search servers are unavailable") from last_error
        return []

    async def nearby(self, lat: float, long: float, radius_km: int = 5) -> tuple[list[dict], int, bool]:
        async def cached_search(radius: int) -> list[dict]:
            key = (round(lat, 3), round(long, 3), radius)
            cached = self._overpass_cache.get(key)
            if cached and cached[0] > time.monotonic():
                return cached[1]
            async with self._overpass_lock:
                cached = self._overpass_cache.get(key)
                if cached and cached[0] > time.monotonic():
                    return cached[1]
                results = await self._fetch_overpass(lat, long, radius * 1000)
                self._overpass_cache[key] = (time.monotonic() + CACHE_TTL_SECONDS, results)
                return results

        places = await cached_search(radius_km)
        widened = len(places) < MIN_LOCAL_RESULTS
        if widened:
            places = await cached_search(15)
            radius_km = 15
        return places, radius_km, widened

    async def search(self, *, lat: float | None = None, long: float | None = None, query: str | None = None) -> dict:
        if query is not None:
            location = await self.geocode(query)
            if location is None:
                return {"location": query, "lat": None, "long": None, "radius_km": 0, "radius_widened": False, "doctors": []}
            lat, long = location.lat, location.long
        if lat is None or long is None:
            raise ValueError("Provide either a city/pincode or both lat and long")
        doctors, radius_km, widened = await self.nearby(lat, long)
        return {"location": query, "lat": lat, "long": long, "radius_km": radius_km, "radius_widened": widened, "doctors": doctors}


finder = OpenStreetMapDoctorFinder()
