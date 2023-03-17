from dataset.impl.sbu_kinect import DatasetSBUKinect
from dataset.impl.sbu_kinect_react import DatasetSBUKinect as DatasetSBUKinect_react
from dataset.impl.sbu_kinect_test import DatasetSBUKinect as DatasetSBUKinect_test
from dataset.impl.sbu_kinect_test_react import DatasetSBUKinect as DatasetSBUKinect_test_react


# ----------------------------------------------------------------------------------------------------------------------
class DataFactory:
    """
    A factory class for instantiating different datasets
    """
    dataset_names = [
            'sbu','sbu_react','sbu_test','sbu_test_react'
        ]

    @staticmethod
    def instantiate(dataset_name, num_synth):
        """
        Instantiates a dataset with its name
        """

        if dataset_name not in DataFactory.dataset_names:
            raise Exception('Unknown dataset "{}"'.format(dataset_name))

        if dataset_name == "sbu":
            return DatasetSBUKinect(num_synth=num_synth)
        if dataset_name == "sbu_test":
            return DatasetSBUKinect_test(num_synth=num_synth)
        if dataset_name == "sbu_test_react":
            return DatasetSBUKinect_test_react(num_synth=num_synth)
        if dataset_name == "sbu_react":
            return DatasetSBUKinect_react(num_synth=num_synth)

        raise Exception('Unknown dataset "{}"'.format(dataset_name))
