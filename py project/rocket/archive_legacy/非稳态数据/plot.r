library(ggplot2)

# --- 0. Global Control Parameter ---
# Adjust this single value to control all font sizes in the plot
base_font_size <- 20

# --- 1. Define Derived Font Sizes ---
# Font sizes for different elements are derived from the base size for consistency
title_font_size <- base_font_size + 6
label_font_size <- base_font_size + 2
tick_font_size <- base_font_size

# --- 2. Data Preparation ---
# Your new data is placed in this data frame
df <- data.frame(
  Time = c(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
  MaxAltitude = c(267, 322, 366, 383, 390, 393, 395, 395, 395, 394, 317, 314, 310, 307, 304, 300, 297, 297, 290, 287, 284)
)

# --- 3. Plotting and Styling ---

# Create the plot object
plot <- ggplot(df, aes(x = Time, y = MaxAltitude)) +
  # a) Add the line and point layers
  geom_line(color = "#2a9d8f", linewidth = 0.75) +
  geom_point(color = "#2a9d8f", size = 2) +
  
  # b) Set title and axis labels
  labs(
    title = "",
    x = "nose cone length", # Updated label for the new data
    y = "Max Flight Hight"
  ) +
  
  # c) Apply a custom theme to replicate the minimalist style
  theme(
    # Set white background for plot and panel
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white"),
    
    # Remove panel grid lines
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    # Keep only bottom and left axis lines
    axis.line.x = element_line(color = "black", linewidth = 0.6),
    axis.line.y = element_line(color = "black", linewidth = 0.6),
    
    # Style the text elements using the font size variables
    plot.title = element_text(size = title_font_size, face = "bold", hjust = 0.5), # hjust=0.5 centers title
    axis.title.x = element_text(size = label_font_size, face = "bold"),
    
    # Style the Y axis title to be horizontal
    axis.title.y = element_text(size = label_font_size, face = "bold", angle = 90, vjust = 0.5), # angle=0 is horizontal
    
    # Style the axis tick labels
    axis.text = element_text(size = tick_font_size)
  )

# d) Display the plot
print(plot)

















# Load the ggplot2 library
library(ggplot2)

# --- 0. Global Control Parameter ---
# Adjust this single value to control all font sizes in the plot
base_font_size <- 20

# --- 1. Define Derived Font Sizes ---
# Font sizes for different elements are derived from the base size for consistency
title_font_size <- base_font_size + 6
label_font_size <- base_font_size + 2
tick_font_size <- base_font_size

# --- 2. Data Loading from CSV ---
# Define the path to your data file
file_path <- "LHS.csv"

# Read the CSV file. 
# We use `header = FALSE` because we assume the CSV has no column names.
# If your CSV file *does* have a header row, change this to `header = TRUE`.
df <- read.csv(file_path, header = FALSE)

# Assign column names for plotting.
# V1 (first column) becomes BaseDiameter, V2 (second column) becomes MaxAltitude.
colnames(df) <- c("BaseDiameter", "MaxAltitude")

# --- 3. Plotting and Styling ---

# Create the plot object using the data from the CSV
plot <- ggplot(df, aes(x = BaseDiameter, y = MaxAltitude)) +
  # a) Add the line and point layers
  geom_line(color = "#2a9d8f", linewidth = 0.75) +
  geom_point(color = "#2a9d8f", size = 2) +
  
  # b) Set title and axis labels
  labs(
    title = "",
    x = "Time",
    y = "Flight Hight"
  ) +
  
  # c) Apply a custom theme to replicate the minimalist style
  theme(
    # Set white background for plot and panel
    plot.background = element_rect(fill = "white", color = NA),
    panel.background = element_rect(fill = "white"),
    
    # Remove panel grid lines
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    
    # Keep only bottom and left axis lines
    axis.line.x = element_line(color = "black", linewidth = 0.6),
    axis.line.y = element_line(color = "black", linewidth = 0.6),
    
    # Style the text elements using the font size variables
    plot.title = element_text(size = title_font_size, face = "bold", hjust = 0.5), # hjust=0.5 centers title
    axis.title.x = element_text(size = label_font_size, face = "bold"),
    
    # Style the Y axis title to be horizontal
    axis.title.y = element_text(size = label_font_size, face = "bold", angle = 90, vjust = 0.5), # angle=0 is horizontal
    
    # Style the axis tick labels
    axis.text = element_text(size = tick_font_size)
  )

# d) Display the plot
print(plot)

