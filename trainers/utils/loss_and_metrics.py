import keras


def get_loss_and_metrics(label_count: int, custom_metrics: dict, with_sample_weights: bool):
    if label_count > 2:
        loss, raw_metrics, not_weighted_metrics = __multiclass(custom_metrics=custom_metrics,
                                                               label_count=label_count)
    elif label_count == 2:
        loss, raw_metrics, not_weighted_metrics = __binary(custom_metrics=custom_metrics)
    else:
        raise ValueError(f"The label count is {label_count}, this is not possible. You need a label cont from 2 "
                         f"(binary) or higher (multiclass)!")

    metrics, weighted_metrics = __fill_metrics(with_sample_weights=with_sample_weights,
                                               raw_metrics=raw_metrics,
                                               not_weighted_metric=not_weighted_metrics)

    return loss, metrics, weighted_metrics


def __binary(custom_metrics: dict):
    loss = keras.losses.BinaryCrossentropy()
    raw_metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy")
    ]

    not_weighted_metric = [
        # add in not_weighted_metric metrics that should not be affected by class_weights or sample_weights
        # For example absolut values, like tp, tn, fp, fn (True Positives, ....)
    ]

    for key in custom_metrics.keys():
        raw_metrics.append(custom_metrics[key]["metric"](**custom_metrics[key]["args"]))

    return loss, raw_metrics, not_weighted_metric


def __multiclass(custom_metrics: dict, label_count: int):
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # TODO, check if from logits?
    raw_metrics = [
        keras.metrics.SparseCategoricalAccuracy(name="accuracy")
    ]

    not_weighted_metric = [
        # add in not_weighted_metric metrics that should not be affected by class_weights or sample_weights
        # For example absolut values, like tp, tn, fp, fn (True Positives, ....)
    ]

    for key in custom_metrics.keys():
        raw_metrics.append(custom_metrics[key]["metric"](
            num_classes=label_count, **custom_metrics[key]["args"]))

    return loss, raw_metrics, not_weighted_metric


def __fill_metrics(with_sample_weights: bool, raw_metrics, not_weighted_metric):
    metrics, weighted_metrics = [], None

    if with_sample_weights:
        weighted_metrics = raw_metrics.copy()
    else:
        metrics = raw_metrics.copy()

    metrics += not_weighted_metric.copy()
    if not metrics:
        metrics = None

    return metrics, weighted_metrics
