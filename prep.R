# SETUP ------------------------------------------------------------------------
if(file.exists(".renv_restored")) renv::load()
suppressPackageStartupMessages(library(tidyverse))
ochim_path <- "data/features/"
set.seed(123)

# READ AND RESHAPE OCHIM FOR PCA -----------------------------------------------
# read data
print("reshaping downloaded data")

df <- paste0(ochim_path, "segmented-500-ms") |> 
  list.files(pattern = "\\.csv$", full.names = TRUE, recursive = TRUE) |> 
  map_dfr(~read_csv(.x, col_types = "ccdcdd", progress = FALSE))

# reshape
df <- df |> 
  mutate(feature = paste(feature, statistic, difference_order, sep = "_")) |> 
  select(track_id, feature, time, data) |> 
  mutate(time = round(time, 1)) |> 
  pivot_wider(names_from = "feature", values_from = data) |> 
  rename_all(~str_replace_all(., "-", "_"))

# trim beginning and end of tracks
df <- df |> 
  mutate_at(vars(starts_with("idyom")), ~replace_na(., 0)) |> 
  group_by(track_id) |> 
  mutate(is_complete = complete.cases(pick(everything()))) |> 
  filter(
    row_number() >= min(which(is_complete)) & 
    row_number() <= max(which(is_complete))
  ) |>
  select(-is_complete) |> 
  ungroup()

# RUN PCA ----------------------------------------------------------------------
print("running pca")

pca <- prcomp(
  formula = ~ ., 
  data = select(df, -track_id, -time),
  center = TRUE, 
  scale. = TRUE, 
  na.action = na.exclude
)

# keep components with eigenvalues > 1
df <- df |> 
  select(track_id, time) |>
  bind_cols(as_tibble(pca$x[, 1:sum(pca$sdev^2 > 1)]))

# ASSIGN K-FOLDS ---------------------------------------------------------------
print("assigning k-folds")

k_folds <- read_csv("data/ochim.csv", show_col_types = FALSE) |> 
  distinct(track_id) |> 
  slice_sample(prop = 1) |> 
  mutate(fold = (row_number() - 1) %% 5 + 1)

# AUGMENT CHILLS ONSETS +- 1 SEC -----------------------------------------------
print("augmenting onsets of chills")

onsets <- read_csv("data/ochim.csv", show_col_types = FALSE) |> 
  select(track_id, chills_id, chills_onset) |> 
  mutate(chills_onset = round(chills_onset, 1)) |> 
  uncount(5) |> 
  group_by(track_id, chills_onset) |> 
  mutate(chills_onset = chills_onset + seq(-1, 1, by = 0.5)) |> 
  filter(chills_onset >= 0) |> 
  ungroup()

# EXPORT LABELLED TRACKS -------------------------------------------------------
print("exporting processed data")

df <- df |> 
  left_join(onsets, by = c("track_id", "time" = "chills_onset")) |>
  left_join(k_folds, by = "track_id") |> 
  mutate(label = ifelse(is.na(chills_id), 0, 1)) |> 
  select(-chills_id) |> 
  select(fold, track_id, time, label, PC1:last_col()) |> 
  group_by(track_id) |> 
  filter(sum(label) != 0) |> 
  ungroup() |> 
  arrange(fold, track_id, time) |> 
  distinct()

# split into 5 files as expected by python script
for (i in 1:5) {
  df |> 
    filter(fold == i) |> 
    write_csv(paste0("data/preprocessed/k", i, ".csv"))
}
