import unittest

from app.services.osm_doctors import Coordinates, OpenStreetMapDoctorFinder


class SparseThenWiderFinder(OpenStreetMapDoctorFinder):
    def __init__(self):
        super().__init__()
        self.radii = []

    async def _fetch_overpass(self, lat, long, radius_m):
        self.radii.append(radius_m)
        if radius_m == 5000:
            return [{"name": "Local Clinic"}, {"name": "Local Doctor"}]
        return [{"name": f"Wider {index}"} for index in range(4)]


class DoctorFinderTests(unittest.IsolatedAsyncioTestCase):
    async def test_overpass_endpoint_failure_falls_back_to_second_public_instance(self):
        service = OpenStreetMapDoctorFinder()
        endpoints = []

        async def fake_fetch(endpoint, lat, long, radius_m):
            endpoints.append(endpoint)
            if len(endpoints) == 1:
                raise TimeoutError("primary busy")
            return [{"name": "Women's clinic"}]

        service._fetch_overpass_from = fake_fetch
        results = await service._fetch_overpass(12.9, 77.5, 5000)
        self.assertEqual(results, [{"name": "Women's clinic"}])
        self.assertGreaterEqual(len(endpoints), 2)

    async def test_sparse_five_kilometer_result_widens_once(self):
        service = SparseThenWiderFinder()
        results, radius, widened = await service.nearby(12.9716, 77.5946)
        self.assertEqual(service.radii, [5000, 15000])
        self.assertEqual(radius, 15)
        self.assertTrue(widened)
        self.assertEqual(len(results), 4)

    async def test_geocode_query_cache_avoids_repeating_nominatim(self):
        service = OpenStreetMapDoctorFinder()
        calls = []
        async def fake_fetch(query):
            calls.append(query)
            return Coordinates(12.9716, 77.5946)
        service._fetch_geocode = fake_fetch
        first = await service.geocode("Bengaluru 560001")
        second = await service.geocode(" bengaluru   560001 ")
        self.assertEqual(first, second)
        self.assertEqual(calls, ["Bengaluru 560001"])

    async def test_overpass_results_cache_by_rounded_coordinates_and_radius(self):
        service = SparseThenWiderFinder()
        calls = []
        async def fake_fetch(lat, long, radius_m):
            calls.append(radius_m)
            return [{"name": f"Result {index}"} for index in range(3)]
        service._fetch_overpass = fake_fetch
        first = await service.nearby(12.97161, 77.59461)
        second = await service.nearby(12.97162, 77.59462)
        self.assertEqual(first, second)
        self.assertEqual(calls, [5000])

    def test_osm_tags_and_missing_phone_are_parsed_safely(self):
        results = OpenStreetMapDoctorFinder._parse_elements({"elements": [{
            "type": "way", "id": 10, "center": {"lat": 12.9, "lon": 77.5},
            "tags": {"name": "Women's Health Clinic", "healthcare:speciality": "gynaecology", "addr:housenumber": "12", "addr:street": "Main Rd", "addr:city": "Mysuru"},
        }]})
        self.assertEqual(results[0]["address"], "12, Main Rd, Mysuru")
        self.assertEqual(results[0]["phone"], "not listed")
        self.assertIn("mlat=12.900000&mlon=77.500000", results[0]["directions_url"])

    def test_general_clinics_and_doctors_are_excluded(self):
        results = OpenStreetMapDoctorFinder._parse_elements({"elements": [
            {"type": "node", "id": 1, "lat": 12.9, "lon": 77.5, "tags": {"name": "General Clinic", "amenity": "clinic"}},
            {"type": "node", "id": 2, "lat": 12.9, "lon": 77.5, "tags": {"name": "Women's Health Clinic", "amenity": "clinic"}},
        ]})
        self.assertEqual([place["name"] for place in results], ["Women's Health Clinic"])

    def test_overpass_query_selects_womens_health_specialties(self):
        query = OpenStreetMapDoctorFinder._overpass_query(12.9, 77.5, 5000)
        self.assertIn("healthcare:speciality", query)
        self.assertIn("gynaecologist", query)
        self.assertNotIn('["amenity"="clinic"](around:', query)


if __name__ == "__main__":
    unittest.main()
