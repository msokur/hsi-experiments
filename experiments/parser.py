class Parser:
    def __init__(self):
        Parser.test_parser()

    @staticmethod
    def parse_combination_parameters(name):
        parts = name.split('_')

        size = parts[1]
        size = int(size[2:])

        scaling = parts[2]
        if scaling[1] == 'l':
            scaling = 'normalization'
        elif scaling[1] == 's':
            scaling = 'svn'

        sample_weights = parts[3]
        if sample_weights[1] == 'T':
            sample_weights = True
        elif sample_weights[1] == 'F':
            sample_weights = False

        smoothing_dimension = parts[4]
        smoothing_dimension = int(smoothing_dimension[1:])

        smoothing = parts[5]
        if len(smoothing) == 1:
            smoothing = 'no'
        else:
            smoothing = smoothing[1]

        smoothing_value = parts[6]
        smoothing_value = float(smoothing_value[1:])

        background = parts[7]
        if background[1] == 'T':
            background = True
        elif background[1] == 'F':
            background = False

        background_blood = parts[8]
        if background:
            background_blood = float(background_blood[1:])
        else:
            background_blood = 0

        background_light = parts[9]
        if background:
            background_light = float(background_light[1:])
        else:
            background_light = 0

        return size, \
               scaling, \
               sample_weights, \
               smoothing_dimension, \
               smoothing, \
               smoothing_value, \
               background, \
               background_blood, \
               background_light

    @staticmethod
    def test_parser():
        results = Parser.parse_combination_parameters('164_3D3_Nl_WF_S1_Sm_S3_BT_B0.1_B0.4_')
        print(results)
        assert results[0] == 3
        assert results[1] == 'normalization'
        assert not results[2]
        assert results[3] == 1
        assert results[4] == 'm'
        assert results[5] == 3
        assert results[6]
        assert results[7] == 0.1
        assert results[8] == 0.4
