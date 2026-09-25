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
#
# TraceLink face-restoration model assets (Phase 8).
#
# The GFPGAN production restorer
# (backend/app/services/gfpgan_adapter.py) expects its weights at:
#
#   backend/app/assets/GFPGANv1.3.pth
#
# or at the path configured via the GFPGAN_MODEL_PATH environment
# variable. The same rules apply: the weights file is intentionally
# NOT committed to the repository (see .gitignore: *.pth) and must
# NEVER be stored in the private evidence bucket (it is model
# infrastructure, not case evidence).
#
# Model: GFPGANv1.3 (TencentARC/GFPGAN v1.3 release). The adapter
# consumes the weights through the gfpgan inference package
# (GFPGANer, upscale=1, no background upsampler), which performs
# its own internal face detection/alignment -- restored outputs
# are therefore recorded with realigned=True and the service uses
# versioned canonical restored-frame geometry for SFace (see
# backend/app/services/face_preparation.py).
#
# Model: GFPGANv1.3, TencentARC/GFPGAN release v1.3.0, downloaded
# from the official release:
#
#   https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth
#
# SHA-256 of the release file (verified at download, 2026-09-24):
#
#   c953a88f2727c85c3d9ae72e2bd4846bbaf59fe6972ad94130e23e7017524a70
#
# File size: 348,632,874 bytes. License terms were NOT verified --
# review the upstream repository license before production use.
#
# The service hashes the loaded file at runtime and stores the
# digest as model_sha256 provenance on every restoration run; the
# hash is never used as an allow-list, so a re-downloaded file
# with the same version but different bytes stays traceable.
#
# Dependency compatibility (verified 2026-09-24 on
# torch==2.9.1+cpu / torchvision==0.24.1+cpu):
# gfpgan==1.3.8, basicsr==1.4.2, facexlib==0.3.0 (see
# backend/requirements-restoration.txt). basicsr imports the
# removed ``torchvision.transforms.functional_tensor`` namespace;
# the adapter supplies it through the local shim in
# backend/app/services/gfpgan_compat.py (no site-packages or
# BasicSR sources modified, no Torch downgrade). Verified end to
# end: GFPGANer construction + enhance() on CPU in
# backend/tests/test_gfpgan_real.py.
#
# Auxiliary weights (REQUIRED at runtime): GFPGANer performs its
# internal face detection/alignment with two facexlib models that
# it downloads from the network when absent. Request-time
# downloads are forbidden: the adapter pre-flights these files
# and fails closed when either is missing. Pre-seed them at the
# facexlib-expected location ``gfpgan/weights/`` relative to the
# process working directory -- when serving from backend/, that is
# backend/gfpgan/weights/ (weights/*.pth are git-ignored, like all
# model files):
#
#   backend/gfpgan/weights/detection_Resnet50_Final.pth
#     RetinaFace-ResNet50 detector, facexlib release v0.1.0:
#     https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
#     SHA-256 (verified at download):
#     6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d
#     File size: 109,497,761 bytes.
#
#   backend/gfpgan/weights/parsing_parsenet.pth
#     ParseNet face parser, facexlib release v0.2.2:
#     https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth
#     SHA-256 (verified at download):
#     3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2
#     File size: 85,331,193 bytes.
#
# The adapter records both identities (file, URL, SHA-256) plus
# the inference device in the run's aux_model_info provenance
# field. Measured on CPU (torch 2.9.1+cpu): GFPGANer construction
# ~6 s (one-time, cached), single-face enhance() ~2-3 s.
#
# Automated tests do not require this file: they use FakeRestorer
# (backend/tests/fake_restorer.py). The real-model test
# (backend/tests/test_gfpgan_real.py) skips automatically when
# the weights or the gfpgan runtime are absent.
