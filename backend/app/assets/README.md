# TraceLink face-detection model assets (Phase 4).
#
# The YuNet production detector (backend/app/services/yunet_detector.py)
# expects its weights at:
#
#   backend/app/assets/face_detection_yunet_2023mar.onnx
#
# or at the path configured via the YUNET_MODEL_PATH environment
# variable. The weights file is intentionally NOT committed to the
# repository and must NEVER be stored in the private evidence bucket
# (it is model infrastructure, not case evidence).
#
# To supply the model locally, download the YuNet ONNX weights from
# the OpenCV Zoo face_detection_yunet release and place the file at
# the path above (or point YUNET_MODEL_PATH at it). Verify the exact
# file hash and license terms at download time before production use.
#
# Automated tests do not require this file: they use FakeDetector
# (backend/tests/fake_detector.py). The real-model test
# (backend/tests/test_yunet_real.py) skips automatically when the
# weights or OpenCV runtime are absent.

# TraceLink face-representation model assets (Phase 5).
#
# The SFace production representation
# (backend/app/services/sface_representation.py) expects its weights
# at:
#
#   backend/app/assets/face_recognition_sface_2021dec.onnx
#
# or at the path configured via the SFACE_MODEL_PATH environment
# variable. The same rules apply: the weights file is intentionally
# NOT committed to the repository and must NEVER be stored in the
# private evidence bucket (it is model infrastructure, not case
# evidence).
#
# Verify the exact file hash before production use. The reference
# SHA-256 of the upstream OpenCV Zoo release is:
#
#   0BA9FBFA01B5270C96627C4EF784DA859931E02F04419C829E83484087C34E79
#
# The service hashes the loaded file at runtime and stores the
# digest as model_sha256 provenance on every embedding; the hash is
# never used as an allow-list, so a re-downloaded file with the
# same version but different bytes stays traceable.
#
# Automated tests do not require this file: they use
# FakeRepresentation (backend/tests/fake_representation.py). The
# real-model tests skip automatically when the weights, the
# PostgreSQL test database, or the OpenCV runtime are absent.

# TraceLink image-enhancement model assets (Phase 7).
#
# The Real-ESRGAN production enhancer
# (backend/app/services/realesrgan_adapter.py) expects its weights
# at:
#
#   backend/app/assets/RealESRGAN_x4plus.pth
#
# or at the path configured via the ENHANCER_MODEL_PATH environment
# variable. The same rules apply: the weights file is intentionally
# NOT committed to the repository (see .gitignore: *.pth) and must
# NEVER be stored in the private evidence bucket (it is model
# infrastructure, not case evidence).
#
# Model: RealESRGAN_x4plus, release v0.1.0, downloaded from the
# official xinntao/Real-ESRGAN release:
#
#   https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
#
# Reference SHA-256 of the release file (verified at download):
#
#   4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1
#
# Architecture: RRDBNet, scale 4, 64 features, 23 blocks (verified
# against the release state dict; the adapter loads it strictly).
# File size: 67,040,989 bytes.
#
# License: Real-ESRGAN code and the x4plus weights are released
# under BSD-3-Clause. No realesrgan/basicsr/gfpgan packages are
# required: the adapter runs the weights directly on torch (CPU
# build by default; see backend/requirements-enhancement.txt),
# because basicsr is incompatible with modern torchvision and
# realesrgan would drag in GFPGAN (an explicit Phase 7 non-goal).
#
# The service hashes the loaded file at runtime and stores the
# digest as model_sha256 provenance on every enhancement run; the
# hash is never used as an allow-list, so a re-downloaded file with
# the same version but different bytes stays traceable.
#
# Automated tests do not require this file: they use FakeEnhancer
# (backend/tests/fake_enhancer.py). The real-model test
# (backend/tests/test_realesrgan_real.py) skips automatically when
# the weights or the torch runtime are absent.
