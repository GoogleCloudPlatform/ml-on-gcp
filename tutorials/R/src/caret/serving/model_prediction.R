# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set the project id
project_id <- "your-project-id"

# Set yout GCS bucket
bucket <- "your_bucket_id"

model_name <- 'caret_babyweight_estimator'
model_dir <- paste0('models/', model_name)
gcs_model_dir <- paste0("gs://", bucket, "/models/", model_name)

print("Downloading model file from GCS...")
command <- paste("gsutil cp -r", gcs_model_dir, ".")
system(command)
print("model files downloaded.")

print("Loading model ...")
model <- readRDS(paste0(model_name,'/trained_model.rds'))
print("Model is loaded.")

#* @post /estimate
estimate_babyweights <- function(req){
  library("rjson")
  instances_json <- req$postBody
  instances <- jsonlite::fromJSON(instances_json)
  df_instances <- data.frame(instances)
  # fix data types
  boolean_columns <- c("is_male", "mother_married", "cigarette_use", "alcohol_use")
  for(col in boolean_columns){
    df_instances[[col]] <- as.logical(df_instances[[col]])
  }

  estimates <- predict(model, df_instances)
  return(estimates)

}





