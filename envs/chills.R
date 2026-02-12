# R VERSION CHECK --------------------------------------------------------------
print("checking R version")
v <- getRversion()
if (v < "4.1.0") stop("you are using R ", v, ", please update to ≥ 4.1.0")

# SET UP ENVIRONMENT -----------------------------------------------------------
print("bootstrapping environment")
if (!dir.exists("renv/library")) {
  # activate local renv
  source("renv/activate.R")
  # restore env
  print("restoring environment")
  renv::restore(prompt = FALSE)
  # install tidyverse wrapper
  if (!requireNamespace("tidyverse", quietly = TRUE)) renv::install("tidyverse")
  # tag env as restored
  invisible(file.create(".renv_restored"))
}
