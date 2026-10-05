def explain(probability: float, risk_label: str) -> str:
    pct = round(probability * 100)
    if risk_label == "elevated":
        return f"Your screening estimate is {pct}% and falls in the elevated range for this model. This is not a diagnosis: the model can’t confirm PCOS or explain symptoms. Please discuss your concerns with a healthcare professional, who can consider your history and appropriate tests."
    return f"Your screening estimate is {pct}% and falls in the lower range for this model. A lower estimate does not rule out PCOS. This is not a diagnosis; if you have concerns or symptoms, please discuss them with a healthcare professional."
