from pydantic import BaseModel, Field

class PulsarFeatures(BaseModel):
    mean_profile: float = Field(..., description="Mean of the integrated pulse profile")
    std_profile: float = Field(..., description="Standard deviation of the integrated pulse profile")
    kurtosis_profile: float = Field(..., description="Excess kurtosis of the integrated pulse profile")
    skewness_profile: float = Field(..., description="Skewness of the integrated pulse profile")
    mean_dm_snr: float = Field(..., description="Mean of the DM-SNR curve")
    std_dm_snr: float = Field(..., description="Standard deviation of the DM-SNR curve")
    kurtosis_dm_snr: float = Field(..., description="Excess kurtosis of the DM-SNR curve")
    skewness_dm_snr: float = Field(..., description="Skewness of the DM-SNR curve")

    class Config:
        json_schema_extra = {
            "example": {
                "mean_profile": 140.5,
                "std_profile": 55.7,
                "kurtosis_profile": -0.2,
                "skewness_profile": 0.3,
                "mean_dm_snr": 3.2,
                "std_dm_snr": 19.1,
                "kurtosis_dm_snr": 7.9,
                "skewness_dm_snr": 74.2,
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = non-pulsar, 1 = pulsar")
    probability: float = Field(..., description="Predicted probability of being a pulsar")
    threshold_used: float