import itertools
import abc
import numpy as np
import run_experiment
# for experiments in run_experiment.py


class Combinations:
    @abc.abstractmethod
    def add_combinations(self, combinations):
        pass


class SmoothingCombinations(Combinations):
    def __init__(self, parameters_for_experiment, combinations_keys, median_params, gaussian_params):
        self.combinations_keys = combinations_keys
        self.median_params = median_params
        self.gaussian_params = gaussian_params
        self.parameters_for_experiment = parameters_for_experiment

    def add_combinations(self, combinations):
        combinations = self.filter_combinations(combinations)
        if 'SMOOTHING_TYPE' in self.parameters_for_experiment and \
                self.parameters_for_experiment['SMOOTHING_TYPE']['add_None']:
            params_without_smoothing = self.parameters_for_experiment.copy()
            del params_without_smoothing['SMOOTHING_TYPE']
            del params_without_smoothing['SMOOTHING_VALUE']
            parameters = run_experiment.Experiment.get_parameters(params_without_smoothing)
            raw_combinations = list(itertools.product(*parameters))
            for combination in raw_combinations:
                combination_ = list(combination)
                combination_.append(None)
                combination_.append(-1)
                combinations.append(combination_)

        return combinations

    def filter_combinations(self, combinations):
        if 'SMOOTHING_TYPE' in self.parameters_for_experiment:
            smoothing_type_index = self.combinations_keys.index('SMOOTHING_TYPE')
            smoothing_value_index = self.combinations_keys.index('SMOOTHING_VALUE')

            filtered_combinations = []
            for combination in combinations:
                if combination[smoothing_type_index] == 'gaussian_filter' and \
                        combination[smoothing_value_index] in self.gaussian_params:
                    filtered_combinations.append(combination)
                if combination[smoothing_type_index] == 'median_filter' and \
                        combination[smoothing_value_index] in self.median_params:
                    filtered_combinations.append(combination)

            return filtered_combinations
        return combinations


class BackgroundCombinations(Combinations):
    def __init__(self, background_parameters):
        self.background_parameters = background_parameters

    def add_combinations(self, combinations):
        if self.background_parameters is not None:
            output_combinations = []
            for combination in combinations:
                new_combinations = [list(combination)] * self._get_length_for_duplication()

                background_combinations = self._get_background_combinations()
                for index, c in enumerate(new_combinations):
                    t = list(c)
                    if self._add_combination_with_False(index, t, background_combinations, output_combinations):
                        continue

                    t.append(True)
                    t.append(background_combinations[index - 1][0])
                    t.append(background_combinations[index - 1][1])
                    output_combinations.append(t)

            return output_combinations
        return combinations

    def _get_length_for_duplication(self):
        length = 0
        if 'BACKGROUND.LIGHT_REFLECTION_THRESHOLD' in self.background_parameters:
            length = len(self.background_parameters['BACKGROUND.LIGHT_REFLECTION_THRESHOLD']['parameters'])

        if 'BACKGROUND.BLOOD_THRESHOLD' in self.background_parameters:
            length = len(self.background_parameters['BACKGROUND.BLOOD_THRESHOLD']['parameters'])

        if 'BACKGROUND.LIGHT_REFLECTION_THRESHOLD' in self.background_parameters \
                and 'BACKGROUND.BLOOD_THRESHOLD' in self.background_parameters:
            length = (len(self.background_parameters['BACKGROUND.LIGHT_REFLECTION_THRESHOLD']['parameters']) *
                      len(self.background_parameters['BACKGROUND.BLOOD_THRESHOLD']['parameters']))

        if False in self.background_parameters['BACKGROUND.WITH_BACKGROUND_EXTRACTION']['parameters']:
            length += 1

        return length

    def _get_background_combinations(self):
        if 'BACKGROUND.LIGHT_REFLECTION_THRESHOLD' in self.background_parameters \
                and 'BACKGROUND.BLOOD_THRESHOLD' in self.background_parameters:
            return list(itertools.product(
                self.background_parameters['BACKGROUND.BLOOD_THRESHOLD']['parameters'],
                self.background_parameters['BACKGROUND.LIGHT_REFLECTION_THRESHOLD']['parameters']))

        if 'BACKGROUND.LIGHT_REFLECTION_THRESHOLD' in self.background_parameters:
            return self.background_parameters['BACKGROUND.LIGHT_REFLECTION_THRESHOLD']['parameters']

        if 'BACKGROUND.BLOOD_THRESHOLD' in self.background_parameters:
            return self.background_parameters['BACKGROUND.BLOOD_THRESHOLD']['parameters']

        raise ValueError('Error! Neither LIGHT_REFLECTION_THRESHOLD nor BLOOD_THRESHOLD are specified!')

    def _add_combination_with_False(self, index, t, background_combinations, output_combinations):
        if False in self.background_parameters['BACKGROUND.WITH_BACKGROUND_EXTRACTION']['parameters'] and \
                index == 0:
            t.append(False)
            t.append(background_combinations[0][0])
            t.append(background_combinations[0][1])
            output_combinations.append(t)
            return True
        return False

    @staticmethod
    def validate_background_params(background_params):
        if background_params:
            # There are only 2 legit values for 'BACKGROUND.WITH_BACKGROUND_EXTRACTION': [True] and [True, False]
            if 'BACKGROUND.WITH_BACKGROUND_EXTRACTION' in background_params:
                with_or_without_background = background_params['BACKGROUND.WITH_BACKGROUND_EXTRACTION']['parameters']
                if len(with_or_without_background) == 0:
                    raise ValueError(f"Error! Empty list were passed for BACKGROUND.WITH_BACKGROUND_EXTRACTION: "
                                     f"{with_or_without_background}")
                if len(with_or_without_background) == 1 and not with_or_without_background[0]:
                    raise ValueError(
                        f"Error! False is passed for WITH_BACKGROUND_EXTRACTION: {with_or_without_background}, "
                        f"which means that there is no need in experiments. Just set it in configs.")
                if len(with_or_without_background) > 2:
                    raise ValueError(f"Error! 'BACKGROUND.WITH_BACKGROUND_EXTRACTION' should be either [False, True] or "
                                     f"[True], but more values were given: {with_or_without_background}")
                if np.array(with_or_without_background).dtype != 'bool':
                    raise ValueError(
                        f"Error! Not boolean values are specified for 'BACKGROUND.WITH_BACKGROUND_EXTRACTION': "
                        f"{with_or_without_background}")
                if len(with_or_without_background) == 2 and len(np.unique(with_or_without_background)) != 2:
                    raise ValueError(f"Error! Duplicate values are passed for BACKGROUND.WITH_BACKGROUND_EXTRACTION: "
                                     f"{with_or_without_background}")

            else:
                raise ValueError("Error! 'BACKGROUND.LIGHT_REFLECTION_THRESHOLD' and/or 'BACKGROUND.BLOOD_THRESHOLD' are specified, "
                       "but 'BACKGROUND.WITH_BACKGROUND_EXTRACTION' is not. Experiments would be redundant.")

        return background_params
