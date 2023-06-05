if [ ! -d resources/ ];
then
    mkdir resources 
fi 

if [ ! -d resources/MedMentions ];
then 
    mkdir resources/MedMentions 
fi 

if [ ! -d resources/BC5CDR ];
then
    mkdir resources/BC5CDR
fi 

python fine_tune_model.py \
    --dataset_name BC5CDR \
    --require_training

# python fine_tune_model.py \
#     --dataset_name MedMentions \
#     --require_training
