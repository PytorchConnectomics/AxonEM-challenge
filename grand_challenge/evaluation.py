import SimpleITK

from evalutils import ClassificationEvaluation
from evalutils.io import SimpleITKLoader
from evalutils.validators import (
    NumberOfCasesValidator, UniquePathIndicesValidator, UniqueImagesValidator
)


class Axonem(ClassificationEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=SimpleITKLoader(),
            validators=(
                NumberOfCasesValidator(num_cases=2),
                UniquePathIndicesValidator(),
                UniqueImagesValidator(),
            ),
        )

    def score_case(self, *, idx, case):
        gt_path = case["path_ground_truth"]
        pred_path = case["path_prediction"]

        # Load the images for this case
        gt = self._file_loader.load_image(gt_path)
        pred = self._file_loader.load_image(pred_path)

        # Check that they're the right images
        if (self._file_loader.hash_image(gt) != case["hash_ground_truth"] or
            self._file_loader.hash_image(pred) != case["hash_prediction"]):
            raise RuntimeError("Images do not match")

        # Cast to the same type
        caster = SimpleITK.CastImageFilter()
        caster.SetOutputPixelType(SimpleITK.sitkUInt8)
        caster.SetNumberOfThreads(1)
        gt = caster.Execute(gt)
        pred = caster.Execute(pred)

        # Score the case
        overlap_measures = SimpleITK.LabelOverlapMeasuresImageFilter()
        overlap_measures.SetNumberOfThreads(1)
        overlap_measures.Execute(gt, pred)

        return {
            'FalseNegativeError': overlap_measures.GetFalseNegativeError(),
            'FalsePositiveError': overlap_measures.GetFalsePositiveError(),
            'MeanOverlap': overlap_measures.GetMeanOverlap(),
            'UnionOverlap': overlap_measures.GetUnionOverlap(),
            'VolumeSimilarity': overlap_measures.GetVolumeSimilarity(),
            'JaccardCoefficient': overlap_measures.GetJaccardCoefficient(),
            'DiceCoefficient': overlap_measures.GetDiceCoefficient(),
            'pred_fname': pred_path.name,
            'gt_fname': gt_path.name,
        }


if __name__ == "__main__":
    Axonem().evaluate()
