import os

from src.path import OUT_DIR
from src.submission import masks_to_submission, submission_to_masks


def test_submission_to_masks():
    submission_filename = os.path.join(OUT_DIR, 'sample_submission.csv')
    masks_dirname = os.path.join(OUT_DIR, 'predictions_sample_submission')
    submission_to_masks(submission_filename, masks_dirname=masks_dirname)


def test_masks_to_submission():
    submission_filename = os.path.join(OUT_DIR, 'sample_submission_test.csv')
    masks_dirname = os.path.join(OUT_DIR, 'predictions_sample_submission')
    masks_filenames = sorted([os.path.join(masks_dirname, fn)
                             for fn in os.listdir(masks_dirname)])
    masks_to_submission(submission_filename, masks_filenames)


if __name__ == '__main__':
    test_submission_to_masks()
    test_masks_to_submission()
