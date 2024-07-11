def get_data_spec():
    data_spec = {}

    data_spec['Propaganda'] = {
        'label2id': {
          "Not Propaganda": 0,
          "Propaganda": 1,
          "Unclear": 2,
          "Not Applicable": 3,
        },
        'id2label': {
          "0": "Not Propaganda",
          "1": "Propaganda",
          "2": "Unclear",
          "3": "Not Applicable",
        },
        'label_names': ["Not Propaganda", "Propaganda", "Unclear", "Not Applicable"],
    }

    data_spec['Bias'] = {

        'label2id': {
          "Unbiased": 0,
          "Biased against Palestine": 1,
          "Biased against Israel": 2,
          "Biased against both Palestine and Israel": 3,
          "Biased against others": 4,
          "Unclear": 5,
          "Not Applicable": 6
        },
        'id2label': {
          "0": "Unbiased",
          "1": "Biased against Palestine",
          "2": "Biased against Israel",
          "3": "Biased against both Palestine and Israel",
          "4": "Biased against others",
          "5": "Unclear",
          "6": "Not Applicable"
        },
        'label_names': [
            "Unbiased",
            "Biased against Palestine",
            "Biased against Israel",
            "Biased against both Palestine and Israel",
            "Biased against others",
            "Unclear",
            "Not Applicable"
        ],
    }

    return data_spec
