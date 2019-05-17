#!/bin/bash
# set -x

# The ratios of the splits between the data sets.  Must add to 1.
ratioTrain=70
ratioValidation=20
ratioTest=10

sourceDir="./flowers"

trainingDir="./data_train"
validationDir="./data_validate"
testDir="./data_test"

# Make sure training/validation/test directories exist
if [[ ! -d ${trainingDir} ]]; then
    mkdir ${trainingDir}
fi
if [[ ! -d ${validationDir} ]]; then
    mkdir ${validationDir}
fi
if [[ ! -d ${testDir} ]]; then
    mkdir ${testDir}
fi

total_train_count=0
total_validation_count=0
total_test_count=0

# Go through the species directories
for classDir in $(find ${sourceDir}/*/ -type d); do
    classDir="${classDir%/}" # remove trailing slash

    class_name="ERROR_NO_class_name"
    if [[ ${classDir} =~ '/' ]]; then
        class_name=${classDir##./*/}
    fi
    if [[ ${class_name} == "ERROR_NO_class_name" ]]; then
        echo "ERROR_NO_class_name for ${classDir}"
        continue
    fi

    num_imgs=$(ls -1q ${classDir}/*.jpg | wc -l) # get the number of images for this species
    echo "The number of images for ${classDir} is ---- ${num_imgs} images"

    # Don't accept species with less than 100 images
    # if [[ ${num_imgs} -lt 100 ]]; then
    #     continue
    # fi

    # Create the directories necessary for each species in train/validate/test directories
    if [[ ! -d ${trainingDir}/${class_name} ]]; then
        mkdir "${trainingDir}/${class_name}"
    fi
    if [[ ! -d ${validationDir}/${class_name} ]]; then
        mkdir "${validationDir}/${class_name}"
    fi
    if [[ ! -d ${testDir}/${class_name} ]]; then
        mkdir "${testDir}/${class_name}"
    fi

    breakpointTrVal=$((($num_imgs*$ratioTrain) / 100))
    breakpointValTest=$((($num_imgs*($ratioTrain+$ratioValidation)) / 100))
    echo "Breakpoint 1: ${breakpointTrVal} ---- Breakpoint 2: ${breakpointValTest}"

    index=1
    for img in $(find ${classDir}/*.jpg); do # go through all images within a species directory
        if [[ ${index} -le ${breakpointTrVal} ]]; then
            # move to training data
            cp ${img} "${trainingDir}/${class_name}/${class_name}.${index}.jpg"
            total_train_count=$((total_train_count + 1))
        elif [[ ${index} -le ${breakpointValTest} ]]; then
            # move to validation data
            cp ${img} "${validationDir}/${class_name}/${class_name}.$((index - breakpointTrVal)).jpg"
            total_validation_count=$((total_validation_count + 1))
        else
            # move to test data
            cp ${img} "${testDir}/${class_name}/${class_name}.$((index - breakpointValTest)).jpg"
            total_test_count=$((total_test_count + 1))
        fi
        index=$((index + 1))
    done
done

echo "Total training images: ${total_train_count}"
echo "Total validation images: ${total_validation_count}"
echo "Total test images: ${total_test_count}"
