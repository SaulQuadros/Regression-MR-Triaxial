from .base_model import BaseModel
from .polynomial import PolynomialModel
from .composite_power import CompositePowerModel
from .camila_batch1 import DunlapModel, HicksModel, UzanModel, JohnsonModel, WitczakUzan1988Model, TamBrownModel
from .camila_batch2 import HopkinsModel, NiModel, NCHRP1_28AModel, NCHRP1_37AModel, Ooi1Model
from .camila_batch3 import Witczak1981Model, Pezo1993Model

# Ooi et al. (2) (2004) é igual ao NCHRP 1-37A conforme o texto
class Ooi2Model(NCHRP1_37AModel):
    @property
    def name(self):
        return "Ooi et al. (2) (2004)"

# Mapeamento para facilitar o carregamento no Streamlit
MODELS_MAP = {
    "Dunlap (1963)": DunlapModel,
    "Hicks (1970)": HicksModel,
    "Witczak (1981)": Witczak1981Model,
    "Uzan (1985)": UzanModel,
    "Johnson et al. (1986)": JohnsonModel,
    "Witczak e Uzan (1988)": WitczakUzan1988Model,
    "Tam e Brown (1988)": TamBrownModel,
    "Pezo (1993)": Pezo1993Model,
    "Hopkins et al. (2001)": HopkinsModel,
    "Ni et al. (2002)": NiModel,
    "NCHRP 1-28A (2004)": NCHRP1_28AModel,
    "NCHRP 1-37A (2004)": NCHRP1_37AModel,
    "Ooi et al. (1) (2004)": Ooi1Model,
    "Ooi et al. (2) (2004)": Ooi2Model,
    "Polinomial c/ Intercepto": lambda: PolynomialModel(include_intercept=True),
    "Polinomial s/Intercepto": lambda: PolynomialModel(include_intercept=False),
    "Potência Composta (Genérico)": CompositePowerModel
}
