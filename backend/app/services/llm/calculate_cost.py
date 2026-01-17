from ._pricing_information import model_pricing


def calculate_cost(
    model_name: str,
    input_token: int | None = None,
    output_token: int | None = None,
    formatted: bool = False,
) -> tuple[float | str, float | str, float | str]:
    model_name = model_name.strip().lower()

    pricing_info = model_pricing[model_name]
    if pricing_info is None:
        raise ValueError(f"Model {model_name} does not have a pricing information.")

    input_cost = (
        pricing_info["input_price"] * input_token if input_token else 0
    ) / 1000000
    output_cost = (
        pricing_info["output_price"] * output_token if output_token else 0
    ) / 1000000
    total_cost = input_cost + output_cost

    if formatted:
        input_cost = f"{input_cost:.10f}".rstrip("0").rstrip(".")
        output_cost = f"{output_cost:.10f}".rstrip("0").rstrip(".")
        total_cost = f"{total_cost:.10f}".rstrip("0").rstrip(".")

    return input_cost, output_cost, total_cost
