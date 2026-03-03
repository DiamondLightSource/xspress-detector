#!/bin/bash
# Script to be called by .gitlab-ci.yml to perform container build using gitlab kubernetes executor
# this script is designed for projects below the namespace <group name>/containers
# and deploys images to:
#      $PROJECT_PATH in the work registry for untagged commits
#      $PROJECT_PATH in the prod registry for tagged commits
# requires that CI_WORK_REGISTRY and CI_PROD_REGISTRY are set to a root path
# in your image registry for each of work and production images.
# See Settings->CI->Variables in the gitlab group <group name>/containers

echo 'Building image...'
GROUP=${CI_PROJECT_NAMESPACE%%/containers*}
PROJECT_PATH=${CI_PROJECT_NAMESPACE##*containers/}

if [ -z "${CI_COMMIT_TAG}" ]; then
  CI_VERSION_TAG=$CI_COMMIT_REF_NAME
else
  CI_VERSION_TAG=$CI_COMMIT_TAG
fi

# For building in DLS' gitlab
add_env="OPENSSL_FORCE_FIPS_MODE=0"
sed -i "/^FROM.*AS developer/a ENV $add_env" Dockerfile

CMDROOT="/kaniko/executor --context $CI_PROJECT_DIR"
CMD=$CMDROOT"  --target ${TARGET}"
CMD=$CMD" --destination $CI_REGISTRY_IMAGE/$CI_PROJECT_NAME-${TARGET}:$CI_VERSION_TAG"

(
    set -x
    $CMD
)
