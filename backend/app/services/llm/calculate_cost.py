from ._pricing_information import model_pricing


def calculate_cost(
    model_name: str, input_token: int, output_token: int
) -> tuple[float, float, float]:
    model_name = model_name.strip().lower()

    pricing_info = model_pricing[model_name]
    if pricing_info is None:
        raise ValueError(f"Model {model_name} does not have a pricing information.")

    input_cost = pricing_info["input_price"] * input_token
    output_cost = pricing_info["output_price"] * output_token
    total_cost = input_cost + output_cost
    return float(input_cost), float(output_cost), float(total_cost)
