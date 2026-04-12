from ..geo_unilora.model import GeoUniLoRAModel


class IGUUniLoRAModel(GeoUniLoRAModel):
    """
    IGU-inspired UniLoRA tuner.

    v1 inherits Geo-UniLoRA module replacement and bank/index assignment; training
    scripts provide IGU-driven rank maps during config construction.
    """

    prefix: str = "igu_unilora_"

