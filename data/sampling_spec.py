def get_sampling_spec():
    label_samples = {
        "Propaganda": {
            'Not Propaganda': 200,
            'Propaganda': 107,
            'Unclear': 40,
            'Not Applicable': 17,
        },
        "Bias": {
            'Biased against both Palestine and Israel': 2,
            'Unclear': 5,
            'Biased against others': 5,
            'Not Applicable': 8,
            'Biased against Israel': 13,
            'Biased against Palestine': 22,
            'Unbiased': 159
        }
    }

    return label_samples
