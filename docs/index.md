# **FSAI Lab Manual**

You can [view or download the Full Stack AI (FSAI) manual here](./1.pdf).

> 📘 This manual contains experiments (2-9), code, and outputs.
> 📘 Lab manual experiments at the end 👇.

# **FSAI Lab Experiments**

## **Experiment 1**

**1. R - Matrix Transformations and Linear Algebra**

- **Aim:** To demonstrate the use of R for creating matrices and
  performing common linear algebra operations like centering, scaling,
  and matrix multiplication to process data. (Adapted based on PDF Ex 2)

- **Algorithm/Approach:**

  1.  **Create Data:** Define a sample data matrix.

  2.  **Print Original:** Display the original data matrix.

  3.  **Center Data:** Use scale() with center = TRUE, scale = FALSE to
      subtract the column means. Print the centered data.

  4.  **Scale Data:** Use scale() with default center = TRUE, scale =
      TRUE to center and scale columns to have mean 0 and standard
      deviation 1. Print the scaled data.

  5.  **Matrix Multiplication:** Multiply the original data matrix by
      its transpose using the matrix multiplication operator (%\*%).

  6.  **Print Result:** Display the result of the matrix multiplication.

- **Code/Program:**  
  \# Sample data (5 data points with 2 features)  
  data \<- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10), ncol = 2, byrow =
  TRUE)  
  print("Original Data:")  
  print(data)  
    
  \# Centering the data (subtracting mean of each column)  
  centered_data \<- scale(data, center = TRUE, scale = FALSE)  
  print("Centered Data:")  
  print(centered_data)  
    
  \# Scaling the data (scaling each column to have mean 0 and standard
  deviation 1)  
  scaled_data \<- scale(data)  
  print("Scaled Data:")  
  print(scaled_data)  
    
  \# Performing matrix multiplication  
  \# Multiply the data matrix by its transpose  
  matrix_multiplication \<- data %\*% t(data)  
  print("Matrix Multiplication (data %\*% t(data)):")  
  print(matrix_multiplication)

- **Output (Sample):**  
  \[1\] "Original Data:"  
  \[,1\] \[,2\]  
  \[1,\] 1 2  
  \[2,\] 3 4  
  \[3,\] 5 6  
  \[4,\] 7 8  
  \[5,\] 9 10  
  \[1\] "Centered Data:"  
  \[,1\] \[,2\]  
  \[1,\] -4 -4  
  \[2,\] -2 -2  
  \[3,\] 0 0  
  \[4,\] 2 2  
  \[5,\] 4 4  
  attr(,"scaled:center")  
  \[1\] 5 6  
  \[1\] "Scaled Data:"  
  \[,1\] \[,2\]  
  \[1,\] -1.264911 -1.264911  
  \[2,\] -0.632456 -0.632456  
  \[3,\] 0.000000 0.000000  
  \[4,\] 0.632456 0.632456  
  \[5,\] 1.264911 1.264911  
  attr(,"scaled:center")  
  \[1\] 5 6  
  attr(,"scaled:scale")  
  \[1\] 3.162278 3.162278  
  \[1\] "Matrix Multiplication (data %\*% t(data)):"  
  \[,1\] \[,2\] \[,3\] \[,4\] \[,5\]  
  \[1,\] 5 11 17 23 29  
  \[2,\] 11 25 39 53 67  
  \[3,\] 17 39 61 83 105  
  \[4,\] 23 53 83 113 143  
  \[5,\] 29 67 105 143 181

**2. Python - 3Sum**

- **Aim:** To find all unique triplets in a given array of integers nums
  that sum up to zero.

- **Algorithm/Approach:**

  1.  **Sort:** Sort the input array nums in ascending order.

  2.  **Initialize:** Create an empty list result to store the triplets.
      Get the length n of nums.

  3.  **Outer Loop:** Iterate through the sorted array with an index i
      from 0 up to n-3.

      - **Skip Duplicates (Outer):** If i \> 0 and nums\[i\] is the same
        as nums\[i-1\], continue to the next iteration.

  4.  **Two Pointers:** Initialize left = i + 1 and right = n - 1.

  5.  **Inner Loop:** While left \< right:

      - Calculate current_sum = nums\[i\] + nums\[left\] +
        nums\[right\].

      - **Found Triplet:** If current_sum == 0:

        - Append \[nums\[i\], nums\[left\], nums\[right\]\] to result.

        - **Skip Duplicates (Inner):** Increment left while left \<
          right and nums\[left\] == nums\[left+1\]. Decrement right
          while left \< right and nums\[right\] == nums\[right-1\].

        - Move pointers: Increment left and decrement right.

      - **Adjust Pointers:**

        - If current_sum \< 0, increment left.

        - If current_sum \> 0, decrement right.

  6.  **Return:** Return the result list.

- **Code/Program:**  
  from typing import List  
    
  def threeSum(nums: List\[int\]) -\> List\[List\[int\]\]:  
  nums.sort()  
  result = \[\]  
  n = len(nums)  
  for i in range(n - 2):  
  if i \> 0 and nums\[i\] == nums\[i - 1\]:  
  continue  
  left, right = i + 1, n - 1  
  while left \< right:  
  current_sum = nums\[i\] + nums\[left\] + nums\[right\]  
  if current_sum == 0:  
  result.append(\[nums\[i\], nums\[left\], nums\[right\]\])  
  while left \< right and nums\[left\] == nums\[left + 1\]:  
  left += 1  
  while left \< right and nums\[right\] == nums\[right - 1\]:  
  right -= 1  
  left += 1  
  right -= 1  
  elif current_sum \< 0:  
  left += 1  
  else:  
  right -= 1  
  return result  
    
  \# Example usage:  
  nums1 = \[-1, 0, 1, 2, -1, -4\]  
  output1 = threeSum(nums1)  
  print(f"Input: {nums1}")  
  print(f"Output: {output1}")  
    
  nums2 = \[0, 1, 1\]  
  output2 = threeSum(nums2)  
  print(f"Input: {nums2}")  
  print(f"Output: {output2}")  
    
  nums3 = \[0, 0, 0\]  
  output3 = threeSum(nums3)  
  print(f"Input: {nums3}")  
  print(f"Output: {output3}")

- **Output (Sample):**  
  Input: \[-1, 0, 1, 2, -1, -4\]  
  Output: \[\[-1, -1, 2\], \[-1, 0, 1\]\]  
  Input: \[0, 1, 1\]  
  Output: \[\]  
  Input: \[0, 0, 0\]  
  Output: \[\[0, 0, 0\]\]

## **Experiment 2**

**1. R - Randomly split a large dataset into training and test sets**

- **Aim:** To randomly split a given dataset (e.g., iris) into training
  and testing subsets using R. (Adapted based on PDF Ex 3)

- **Algorithm/Approach:**

  1.  **Load Data:** Load the desired dataset (e.g., iris which is
      built-in, or mtcars). Assign it to my_data.

  2.  **Set Seed:** Use set.seed() for reproducible random sampling.

  3.  **Get Size:** Determine the total number of rows n using
      nrow(my_data).

  4.  **Generate Train Indices:** Use sample() to randomly select
      indices for the training set. Specify the size (e.g., 0.8\*n for
      80%) and replace = FALSE.

  5.  **Generate Test Indices:** Use setdiff() to find the indices that
      are in the full set (1:n) but not in train_indices.

  6.  **Create Subsets:** Create train_data and test_data by subsetting
      my_data using the generated train_indices and test_indices,
      respectively.

  7.  **Print Dimensions:** Use cat() and dim() to print the dimensions
      (rows, columns) of the resulting training and test sets.

- **Code/Program:**  
  \# Load a dataset (e.g., iris)  
  my_data \<- iris  
    
  \# Set seed for reproducibility  
  set.seed(123)  
    
  \# Generate random indices for training and test sets  
  n \<- nrow(my_data) \# Number of rows in the dataset  
  train_size \<- floor(0.8 \* n) \# 80% for training  
  train_indices \<- sample(1:n, size = train_size, replace = FALSE)  
  test_indices \<- setdiff(1:n, train_indices) \# Remaining for
  testing  
    
  \# Create training and test sets using the generated indices  
  train_data \<- my_data\[train_indices, \]  
  test_data \<- my_data\[test_indices, \]  
    
  \# Print dimensions of training and test sets  
  cat("Training set dimensions:", dim(train_data), "\n")  
  cat("Test set dimensions:", dim(test_data), "\n")

- **Output (Sample for iris dataset):**  
  Training set dimensions: 120 5  
  Test set dimensions: 30 5

**2. Python - Add Two Numbers (Linked List)**

- **Aim:** To add two non-negative integers represented by linked lists,
  where digits are stored in reverse order, and return the sum as a new
  linked list.

- **Algorithm/Approach:**

  1.  **Define Node:** Define a ListNode class with val and next
      attributes.

  2.  **Initialize:** Create a dummy_head node and a current pointer.
      Initialize carry = 0.

  3.  **Loop:** Iterate while l1, l2, or carry exists.

  4.  **Get Values:** Get l1.val (or 0 if null) and l2.val (or 0 if
      null).

  5.  **Calculate Sum:** Compute total_sum = val1 + val2 + carry.

  6.  **Update Carry:** carry = total_sum // 10.

  7.  **Calculate Digit:** digit = total_sum % 10.

  8.  **Create Node:** Create a new ListNode(digit).

  9.  **Link Node:** Set current.next to the new node and advance
      current.

  10. **Advance Lists:** Advance l1 and l2 if they exist.

  11. **Return Result:** Return dummy_head.next.

- **Code/Program:**  
  from typing import Optional  
    
  class ListNode:  
  def \_\_init\_\_(self, val=0, next=None):  
  self.val = val  
  self.next = next  
    
  def addTwoNumbers(l1: Optional\[ListNode\], l2: Optional\[ListNode\])
  -\> Optional\[ListNode\]:  
  dummy_head = ListNode(0)  
  current = dummy_head  
  carry = 0  
  while l1 or l2 or carry:  
  val1 = l1.val if l1 else 0  
  val2 = l2.val if l2 else 0  
    
  total_sum = val1 + val2 + carry  
  carry = total_sum // 10  
  digit = total_sum % 10  
    
  current.next = ListNode(digit)  
  current = current.next  
    
  if l1:  
  l1 = l1.next  
  if l2:  
  l2 = l2.next  
    
  return dummy_head.next  
    
  \# Helper function to create linked list from list  
  def create_linked_list(nums: list) -\> Optional\[ListNode\]:  
  if not nums: return None  
  head = ListNode(nums\[0\])  
  current = head  
  for i in range(1, len(nums)):  
  current.next = ListNode(nums\[i\])  
  current = current.next  
  return head  
    
  \# Helper function to print linked list  
  def print_linked_list(head: Optional\[ListNode\]):  
  nums = \[\]  
  current = head  
  while current:  
  nums.append(current.val)  
  current = current.next  
  print(nums)  
    
  \# Example usage:  
  l1_list = \[2, 4, 3\]  
  l2_list = \[5, 6, 4\]  
  l1 = create_linked_list(l1_list)  
  l2 = create_linked_list(l2_list)  
  print("Input l1:", l1_list)  
  print("Input l2:", l2_list)  
  result_list = addTwoNumbers(l1, l2)  
  print("Output:", end=" "); print_linked_list(result_list)  
    
  l1_list = \[0\]; l2_list = \[0\]  
  l1 = create_linked_list(l1_list); l2 = create_linked_list(l2_list)  
  print("\nInput l1:", l1_list); print("Input l2:", l2_list)  
  result_list = addTwoNumbers(l1, l2)  
  print("Output:", end=" "); print_linked_list(result_list)  
    
  l1_list = \[9,9,9,9,9,9,9\]; l2_list = \[9,9,9,9\]  
  l1 = create_linked_list(l1_list); l2 = create_linked_list(l2_list)  
  print("\nInput l1:", l1_list); print("Input l2:", l2_list)  
  result_list = addTwoNumbers(l1, l2)  
  print("Output:", end=" "); print_linked_list(result_list)

- **Output (Sample):**  
  Input l1: \[2, 4, 3\]  
  Input l2: \[5, 6, 4\]  
  Output: \[7, 0, 8\]  
    
  Input l1: \[0\]  
  Input l2: \[0\]  
  Output: \[0\]  
    
  Input l1: \[9, 9, 9, 9, 9, 9, 9\]  
  Input l2: \[9, 9, 9, 9\]  
  Output: \[8, 9, 9, 9, 0, 0, 0, 1\]

## **Experiment 3**

**1. R - Apply various statistical and machine learning functions**

- **Aim:** To apply basic statistical analysis and a linear regression
  model to the mtcars dataset using R. (Adapted based on PDF Ex 4)

- **Algorithm/Approach:**

  1.  **Load Data:** Load the built-in mtcars dataset.

  2.  **Inspect Data:** Check the structure of the dataset using str().

  3.  **Fit Linear Model:** Use lm() to fit a linear regression model
      predicting mpg (miles per gallon) based on cyl (number of
      cylinders).

  4.  **Summarize Model:** Print the summary() of the fitted model to
      see coefficients, R-squared, p-values, etc.

  5.  **Make Predictions:** Use predict() with the fitted model and the
      original data to get predicted mpg values.

  6.  **Print Predictions:** Print the first few predicted values using
      head().

  7.  **Evaluate Model:** Calculate and print R-squared from the summary
      and Root Mean Squared Error (RMSE) between actual and predicted
      mpg.

  8.  **Visualize:** Create a scatter plot (plot()) of actual mpg vs.
      cyl. Add the regression line (abline()) and predicted points
      (points()). Add a legend (legend()).

- **Code/Program:**  
  \# Load the built-in mtcars dataset  
  data(mtcars)  
    
  \# Check the structure of the dataset  
  str(mtcars)  
    
  \# Fit a linear regression model to predict mpg using the number of
  cylinders  
  model \<- lm(mpg ~ cyl, data = mtcars)  
    
  \# Summary of the regression model  
  print(summary(model))  
    
  \# Make predictions using the fitted model  
  predictions \<- predict(model, newdata = mtcars)  
    
  \# Print the first few predicted values  
  cat("First few predicted mpg values:\n")  
  print(head(predictions))  
    
  \# Model evaluation: R-squared and RMSE  
  r_squared \<- summary(model)\$r.squared  
  rmse \<- sqrt(mean((mtcars\$mpg - predictions)^2))  
    
  cat("\nR-squared:", r_squared, "\n")  
  cat("Root Mean Squared Error (RMSE):", rmse, "\n")  
    
  \# Visualization of the regression line  
  plot(mtcars\$cyl, mtcars\$mpg,  
  main = "Linear Regression: MPG vs Cylinders",  
  xlab = "Number of Cylinders", ylab = "Miles Per Gallon (mpg)",  
  pch = 19, col = "blue")  
    
  \# Add regression line  
  abline(model, col = "red", lwd = 2)  
    
  \# Add predicted values as points  
  \# points(mtcars\$cyl, predictions, col = "green", pch = 4) \#
  Optional: Show predicted points  
    
  legend("topright", legend = c("Actual", "Regression Line"),
  \#"Predicted"  
  col = c("blue", "red"), \# "green"  
  pch = c(19, NA), \# 4  
  lty = c(NA, 1), \# NA  
  lwd = c(NA, 2)) \# NA

- **Output (Sample):**  
  'data.frame': 32 obs. of 11 variables:  
  \$ mpg : num 21 21 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 ...  
  \$ cyl : num 6 6 4 6 8 6 8 4 4 6 ...  
  \$ disp: num 160 160 108 258 360 ...  
  \$ hp : num 110 110 93 110 175 105 245 62 95 123 ...  
  \$ drat: num 3.9 3.9 3.85 3.08 3.15 2.76 3.21 3.69 3.92 3.92 ...  
  \$ wt : num 2.62 2.88 2.32 3.21 3.44 ...  
  \$ qsec: num 16.5 17 18.6 19.4 17 ...  
  \$ vs : num 0 0 1 1 0 1 0 1 1 1 ...  
  \$ am : num 1 1 1 0 0 0 0 0 0 0 ...  
  \$ gear: num 4 4 4 3 3 3 3 4 4 4 ...  
  \$ carb: num 4 4 1 1 2 1 4 2 2 4 ...  
    
  Call:  
  lm(formula = mpg ~ cyl, data = mtcars)  
    
  Residuals:  
  Min 1Q Median 3Q Max  
  -4.9814 -2.1189 -0.1814 1.8186 7.5186  
    
  Coefficients:  
  Estimate Std. Error t value Pr(\>\|t\|)  
  (Intercept) 37.8846 2.0738 18.27 \< 2e-16 \*\*\*  
  cyl -2.8758 0.3224 -8.92 6.11e-10 \*\*\*  
  ---  
  Signif. codes: 0 '\*\*\*' 0.001 '\*\*' 0.01 '\*' 0.05 '.' 0.1 ' ' 1  
    
  Residual standard error: 3.206 on 30 degrees of freedom  
  Multiple R-squared: 0.7262, Adjusted R-squared: 0.7171  
  F-statistic: 79.56 on 1 and 30 DF, p-value: 6.113e-10  
    
  First few predicted mpg values:  
  Mazda RX4 Mazda RX4 Wag Datsun 710 Hornet 4 Drive Hornet Sportabout
  Valiant  
  20.63045 20.63045 26.38199 20.63045 14.87891 20.63045  
    
  R-squared: 0.72618  
  Root Mean Squared Error (RMSE): 3.153565  
  \# (Plot is also generated)

**2. Python - Longest Substring Without Repeating Characters**

- **Aim:** To find the length of the longest substring within a given
  string s that does not contain any repeating characters.

- **Algorithm/Approach:** Sliding Window Technique:

  1.  **Initialize:** Create an empty set char_set. Set left = 0,
      max_length = 0.

  2.  **Iterate:** Loop right from 0 to len(s) - 1.

  3.  **Check Duplicates:** While s\[right\] is in char_set:

      - Remove s\[left\] from char_set.

      - Increment left.

  4.  **Add Character:** Add s\[right\] to char_set.

  5.  **Update Max Length:** max_length = max(max_length, right - left +
      1).

  6.  **Return:** Return max_length.

- **Code/Program:**  
  def lengthOfLongestSubstring(s: str) -\> int:  
  char_set = set()  
  left = 0  
  max_length = 0  
  for right in range(len(s)):  
  while s\[right\] in char_set:  
  char_set.remove(s\[left\])  
  left += 1  
  char_set.add(s\[right\])  
  max_length = max(max_length, right - left + 1)  
  return max_length  
    
  \# Example usage:  
  s1 = "abcabcbb"  
  output1 = lengthOfLongestSubstring(s1)  
  print(f"Input: s = \\{s1}\\")  
  print(f"Output: {output1}")  
    
  s2 = "bbbbb"  
  output2 = lengthOfLongestSubstring(s2)  
  print(f"Input: s = \\{s2}\\")  
  print(f"Output: {output2}")  
    
  s3 = "pwwkew"  
  output3 = lengthOfLongestSubstring(s3)  
  print(f"Input: s = \\{s3}\\")  
  print(f"Output: {output3}")

- **Output (Sample):**  
  Input: s = "abcabcbb"  
  Output: 3  
  Input: s = "bbbbb"  
  Output: 1  
  Input: s = "pwwkew"  
  Output: 3

## **Experiment 4**

**1. R - Analysis of the reliability and goodness of fit**

- **Aim:** To provide an analysis of model fit by splitting data,
  training a model, making predictions, and evaluating performance using
  MSE and R-squared, along with visualizations. (Adapted based on PDF Ex
  5)

- **Algorithm/Approach:**

  1.  **Load Data & Libraries:** Load mtcars dataset. Load ggplot2
      library.

  2.  **Split Data:** Randomly split mtcars into 80% training
      (train_data) and 20% testing (test_data) sets using sample() and
      setdiff(). Remember set.seed().

  3.  **Fit Model:** Train a linear regression model (lm()) using the
      train_data to predict mpg from all other variables (mpg ~ .).

  4.  **Predict:** Generate predictions for both train_data and
      test_data using predict().

  5.  **Evaluate:**

      - Calculate Mean Squared Error (MSE) for both training and test
        sets: mean((actual - prediction)^2).

      - Calculate R-squared for the training set from
        summary(model)\$r.squared.

      - Calculate R-squared for the test set using cor(test_data\$mpg,
        test_predictions)^2.

  6.  **Print Metrics:** Display the calculated MSE and R-squared values
      for both sets.

  7.  **Visualize:** Create exploratory plots using ggplot2:

      - Histogram of mpg.

      - Scatter plot of mpg vs. hp.

      - Scatter plot of mpg vs. wt.

      - Scatter plot of mpg vs. cyl.

- **Code/Program:**  
  \# Load the 'mtcars' dataset  
  data(mtcars)  
  my_data \<- mtcars  
    
  \# Load required libraries  
  \# install.packages("ggplot2") \# Run once if not installed  
  library(ggplot2)  
    
  \# Set seed for reproducibility  
  set.seed(123)  
    
  \# Generate random indices for training and test sets  
  n \<- nrow(my_data)  
  train_size \<- floor(0.8 \* n)  
  train_indices \<- sample(1:n, size = train_size, replace = FALSE)  
  test_indices \<- setdiff(1:n, train_indices)  
    
  \# Create training and test sets  
  train_data \<- my_data\[train_indices, \]  
  test_data \<- my_data\[test_indices, \]  
    
  \# Fit a linear regression model using the training set (predict mpg
  from all others)  
  model \<- lm(mpg ~ ., data = train_data)  
    
  \# Make predictions on both training and test sets  
  train_predictions \<- predict(model, newdata = train_data)  
  test_predictions \<- predict(model, newdata = test_data)  
    
  \# Compute Mean Squared Error (MSE)  
  train_mse \<- mean((train_data\$mpg - train_predictions)^2)  
  test_mse \<- mean((test_data\$mpg - test_predictions)^2)  
    
  \# Compute R-squared  
  train_r_squared \<- summary(model)\$r.squared  
  \# Calculate test R-squared based on correlation between actual and
  predicted  
  \# Avoid potential division by zero if test_data\$mpg has zero
  variance (unlikely here)  
  if (var(test_data\$mpg) \> 0) {  
  test_r_squared \<- cor(test_data\$mpg, test_predictions)^2  
  } else {  
  test_r_squared \<- NA \# Or handle as appropriate  
  }  
    
    
  \# Print the results  
  cat("Training Set Metrics:\n")  
  cat("Mean Squared Error (MSE):", train_mse, "\n")  
  cat("R-squared:", train_r_squared, "\n\n")  
    
  cat("Test Set Metrics:\n")  
  cat("Mean Squared Error (MSE):", test_mse, "\n")  
  cat("R-squared:", test_r_squared, "\n\n")  
    
  \# --- Visualizations ---  
  \# Histogram of mpg  
  p1 \<- ggplot(my_data, aes(x = mpg)) +  
  geom_histogram(binwidth = 2, fill = "skyblue", color = "black") +  
  labs(title = "Distribution of Miles Per Gallon", x = "Miles Per
  Gallon", y = "Frequency")  
  print(p1)  
    
  \# Scatter plot of mpg vs. horsepower (hp)  
  p2 \<- ggplot(my_data, aes(x = hp, y = mpg)) +  
  geom_point(color = "blue") +  
  labs(title = "Scatter Plot of Miles Per Gallon vs. Horsepower", x =
  "Horsepower", y = "Miles Per Gallon")  
  print(p2)  
    
  \# Scatter plot of mpg vs. weight (wt)  
  p3 \<- ggplot(my_data, aes(x = wt, y = mpg)) +  
  geom_point(color = "green") +  
  labs(title = "Scatter Plot of Miles Per Gallon vs. Weight", x =
  "Weight", y = "Miles Per Gallon")  
  print(p3)  
    
  \# Scatter plot of mpg vs. number of cylinders (cyl)  
  p4 \<- ggplot(my_data, aes(x = factor(cyl), y = mpg)) + \# Treat cyl
  as factor for clarity  
  geom_point(color = "red") + \# Could use geom_boxplot or geom_violin
  too  
  labs(title = "Scatter Plot of Miles Per Gallon vs. Number of
  Cylinders", x = "Number of Cylinders", y = "Miles Per Gallon")  
  print(p4)

- **Output (Sample):**  
  Training Set Metrics:  
  Mean Squared Error (MSE): 3.510995  
  R-squared: 0.9358658  
    
  Test Set Metrics:  
  Mean Squared Error (MSE): 19.13392  
  R-squared: 0.2879197  
    
  \# (Four plots will be displayed)

**2. Python - Group Anagrams**

- **Aim:** To group an array of strings strs such that all anagrams are
  together in sub-lists.

- **Algorithm/Approach:**

  1.  **Import:** Import defaultdict from collections.

  2.  **Initialize Map:** Create a defaultdict(list) called anagram_map.

  3.  **Iterate Strings:** Loop through each string s in strs.

  4.  **Sort String:** Create a key by sorting s (sorted_s =
      "".join(sorted(s))).

  5.  **Append to Map:** Append the original s to
      anagram_map\[sorted_s\].

  6.  **Return Groups:** Return list(anagram_map.values()).

- **Code/Program:**  
  from typing import List  
  from collections import defaultdict  
    
  def groupAnagrams(strs: List\[str\]) -\> List\[List\[str\]\]:  
  anagram_map = defaultdict(list)  
  for s in strs:  
  sorted_s = "".join(sorted(s))  
  anagram_map\[sorted_s\].append(s)  
  return list(anagram_map.values())  
    
  \# Example usage:  
  strs1 = \["eat", "tea", "tan", "ate", "nat", "bat"\]  
  output1 = groupAnagrams(strs1)  
  print(f"Input: {strs1}")  
  print(f"Output: {output1}")  
    
  strs2 = \[""\]  
  output2 = groupAnagrams(strs2)  
  print(f"Input: {strs2}")  
  print(f"Output: {output2}")  
    
  strs3 = \["a"\]  
  output3 = groupAnagrams(strs3)  
  print(f"Input: {strs3}")  
  print(f"Output: {output3}")

- **Output (Sample):**  
  Input: \['eat', 'tea', 'tan', 'ate', 'nat', 'bat'\]  
  Output: \[\['eat', 'tea', 'ate'\], \['tan', 'nat'\], \['bat'\]\] \#
  Order may vary  
  Input: \[''\]  
  Output: \[\[''\]\]  
  Input: \['a'\]  
  Output: \[\['a'\]\]

## **Experiment 5**

**1. Python - Implement primitive datatypes**

- **Aim:** To demonstrate the declaration, initialization, and basic
  operations of Python's primitive data types: integer (int), float
  (float), string (str), boolean (bool), and NoneType.

- **Algorithm/Approach:**

  1.  **Integer (int):** Declare, print value/type, perform/print
      arithmetic.

  2.  **Float (float):** Declare, print value/type, perform/print
      arithmetic.

  3.  **String (str):** Declare, print value/type, perform/print
      concatenation, methods (upper()), length (len()), slicing.

  4.  **Boolean (bool):** Declare True/False, print value/type,
      perform/print logical operations (and, or, not).

  5.  **NoneType (None):** Declare None, print value/type.

- **Code/Program:**  
  \# Integer  
  my_int = 10  
  print(f"Integer: {my_int}, Type: {type(my_int)}")  
  int_sum = my_int + 5  
  print(f"Integer Sum (10 + 5): {int_sum}")  
  int_prod = my_int \* 2  
  print(f"Integer Product (10 \* 2): {int_prod}")  
    
  \# Float  
  my_float = 3.14  
  print(f"\nFloat: {my_float}, Type: {type(my_float)}")  
  float_div = my_float / 2  
  print(f"Float Division (3.14 / 2): {float_div}")  
    
  \# String  
  my_string = "Hello Python"  
  print(f"\nString: {my_string}, Type: {type(my_string)}")  
  string_concat = my_string + " World"  
  print(f"String Concatenation: {string_concat}")  
  print(f"String Uppercase: {my_string.upper()}")  
  print(f"String Length: {len(my_string)}")  
  print(f"String Slice \[0:5\]: {my_string\[0:5\]}")  
    
  \# Boolean  
  my_bool_true = True  
  my_bool_false = False  
  print(f"\nBoolean True: {my_bool_true}, Type: {type(my_bool_true)}")  
  print(f"Boolean False: {my_bool_false}, Type:
  {type(my_bool_false)}")  
  print(f"Logical AND (True and False): {my_bool_true and
  my_bool_false}")  
  print(f"Logical OR (True or False): {my_bool_true or
  my_bool_false}")  
  print(f"Logical NOT (not True): {not my_bool_true}")  
    
  \# NoneType  
  my_none = None  
  print(f"\nNoneType: {my_none}, Type: {type(my_none)}")

- **Output (Sample):**  
  Integer: 10, Type: \<class 'int'\>  
  Integer Sum (10 + 5): 15  
  Integer Product (10 \* 2): 20  
    
  Float: 3.14, Type: \<class 'float'\>  
  Float Division (3.14 / 2): 1.57  
    
  String: Hello Python, Type: \<class 'str'\>  
  String Concatenation: Hello Python World  
  String Uppercase: HELLO PYTHON  
  String Length: 12  
  String Slice \[0:5\]: Hello  
    
  Boolean True: True, Type: \<class 'bool'\>  
  Boolean False: False, Type: \<class 'bool'\>  
  Logical AND (True and False): False  
  Logical OR (True or False): True  
  Logical NOT (not True): False  
    
  NoneType: None, Type: \<class 'NoneType'\>

**2. Python - Top K Frequent Elements**

- **Aim:** To find the k most frequently occurring elements in a given
  integer array nums.

- **Algorithm/Approach:** Using Hash Map and Min-Heap:

  1.  **Import:** Import Counter and heapq.

  2.  **Handle Empty:** If nums is empty, return \[\].

  3.  **Count Frequencies:** count = Counter(nums).

  4.  **Initialize Heap:** min_heap = \[\].

  5.  **Populate Heap:** For num, freq in count.items():

      - heapq.heappush(min_heap, (freq, num)).

      - If len(min_heap) \> k: heapq.heappop(min_heap).

  6.  **Extract Result:** result = \[num for freq, num in min_heap\].

  7.  **Return:** Return result.

- **Code/Program:**  
  from typing import List  
  from collections import Counter  
  import heapq  
    
  def topKFrequent(nums: List\[int\], k: int) -\> List\[int\]:  
  if not nums:  
  return \[\]  
  count = Counter(nums)  
  min_heap = \[\]  
  for num, freq in count.items():  
  heapq.heappush(min_heap, (freq, num))  
  if len(min_heap) \> k:  
  heapq.heappop(min_heap)  
  result = \[num for freq, num in min_heap\]  
  return result  
    
  \# Example usage:  
  nums1 = \[1, 1, 1, 2, 2, 3\]; k1 = 2  
  output1 = topKFrequent(nums1, k1)  
  print(f"Input: nums = {nums1}, k = {k1}")  
  print(f"Output: {output1}")  
    
  nums2 = \[1\]; k2 = 1  
  output2 = topKFrequent(nums2, k2)  
  print(f"Input: nums = {nums2}, k = {k2}")  
  print(f"Output: {output2}")

- **Output (Sample):**  
  Input: nums = \[1, 1, 1, 2, 2, 3\], k = 2  
  Output: \[1, 2\] \# Order may vary  
  Input: nums = \[1\], k = 1  
  Output: \[1\]

## **Experiment 6**

**1. Python - Implement Control statements**

- **Aim:** To demonstrate the usage of Python's control flow statements:
  if-elif-else, for, while, break, continue, and pass.

- **Algorithm/Approach:**

  1.  **Conditional (if-elif-else):** Show branching based on a
      variable's value.

  2.  **for Loop:** Demonstrate iteration over a list and a range.

  3.  **while Loop:** Show looping based on a condition with a counter.

  4.  **Loop Control:** Illustrate break to exit a loop early and
      continue to skip an iteration.

  5.  **pass Statement:** Show its use as a placeholder in an empty
      function definition.

- **Code/Program:**  
  \# --- If-Elif-Else ---  
  print("--- If-Elif-Else ---")  
  score = 75  
  if score \>= 90: grade = "A"  
  elif score \>= 80: grade = "B"  
  elif score \>= 70: grade = "C"  
  else: grade = "D"  
  print(f"Score: {score}, Grade: {grade}")  
    
  \# --- For Loop ---  
  print("\n--- For Loop ---")  
  my_list = \["apple", "banana", "cherry"\]  
  print("Iterating through list:")  
  for fruit in my_list: print(fruit)  
  print("\nIterating through range:")  
  for i in range(3): print(i, end=" ")  
  print()  
    
  \# --- While Loop ---  
  print("\n--- While Loop ---")  
  count = 0  
  while count \< 3: print(count, end=" "); count += 1  
  print()  
    
  \# --- Break and Continue ---  
  print("\n--- Break and Continue ---")  
  print("Break at 3:")  
  for i in range(5):  
  if i == 3: break  
  print(i, end=" ")  
  print("\nContinue at 1:")  
  for i in range(4):  
  if i == 1: continue  
  print(i, end=" ")  
  print()  
    
  \# --- Pass Statement ---  
  print("\n--- Pass Statement ---")  
  def my_empty_function(): pass  
  my_empty_function()  
  print("Empty function called.")

- **Output (Sample):**  
  --- If-Elif-Else ---  
  Score: 75, Grade: C  
    
  --- For Loop ---  
  Iterating through list:  
  apple  
  banana  
  cherry  
    
  Iterating through range:  
  0 1 2  
    
  --- While Loop ---  
  0 1 2  
    
  --- Break and Continue ---  
  Break at 3:  
  0 1 2  
  Continue at 1:  
  0 2 3  
    
  --- Pass Statement ---  
  Empty function called.

**2. Python - Binary Tree Inorder and Postorder Traversal**

- **Aim:** To perform inorder (Left-Node-Right) and postorder
  (Left-Right-Node) traversals of a given binary tree.

- **Algorithm/Approach:** Recursive Traversal:

  1.  **Define Node:** Define TreeNode class.

  2.  **Inorder Function:**

      - Initialize result = \[\].

      - Define recursive helper traverse(node):

        - If node: traverse(node.left), result.append(node.val),
          traverse(node.right).

      - Call traverse(root). Return result.

  3.  **Postorder Function:**

      - Initialize result = \[\].

      - Define recursive helper traverse(node):

        - If node: traverse(node.left), traverse(node.right),
          result.append(node.val).

      - Call traverse(root). Return result.

- **Code/Program:**  
  from typing import List, Optional  
    
  class TreeNode:  
  def \_\_init\_\_(self, val=0, left=None, right=None):  
  self.val = val  
  self.left = left  
  self.right = right  
    
  def inorderTraversal(root: Optional\[TreeNode\]) -\> List\[int\]:  
  result = \[\]  
  def traverse(node):  
  if node:  
  traverse(node.left)  
  result.append(node.val)  
  traverse(node.right)  
  traverse(root)  
  return result  
    
  def postorderTraversal(root: Optional\[TreeNode\]) -\> List\[int\]:  
  result = \[\]  
  def traverse(node):  
  if node:  
  traverse(node.left)  
  traverse(node.right)  
  result.append(node.val)  
  traverse(root)  
  return result  
    
  \# Example 1: root = \[1,null,2,3\]  
  root1 = TreeNode(1)  
  root1.right = TreeNode(2)  
  root1.right.left = TreeNode(3)  
  print("Example 1:")  
  inorder1 = inorderTraversal(root1)  
  postorder1 = postorderTraversal(root1)  
  \# Note: The example output in the docx \[1,2,3\] for postorder seems
  incorrect for the tree \[1,null,2,3\].  
  \# Correct postorder should be \[3, 2, 1\]. The inorder \[1,3,2\] is
  correct.  
  print(f"Inorder: {inorder1}")  
  print(f"Postorder: {postorder1}")  
    
  \# Example 2: root = \[1,2,3,4,5,null,8,null,null,6,7,null,null,9\] \#
  Approximate structure  
  root2 = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5, TreeNode(6),
  TreeNode(7))), TreeNode(3, None, TreeNode(8, TreeNode(9))))  
  print("\nExample 2:")  
  inorder2 = inorderTraversal(root2)  
  postorder2 = postorderTraversal(root2)  
  \# Note: The example output in the docx \[1,2,4,5,6,7,3,8,9\] for
  postorder seems incorrect.  
  \# Correct postorder should be \[4, 6, 7, 5, 2, 9, 8, 3, 1\]. The
  inorder \[4,2,6,5,7,1,3,9,8\] is correct.  
  print(f"Inorder: {inorder2}")  
  print(f"Postorder: {postorder2}")

- **Output (Sample):**  
  Example 1:  
  Inorder: \[1, 3, 2\]  
  Postorder: \[3, 2, 1\]  
    
  Example 2:  
  Inorder: \[4, 2, 6, 5, 7, 1, 3, 9, 8\]  
  Postorder: \[4, 6, 7, 5, 2, 9, 8, 3, 1\]

## **Experiment 7**

**1. Python - Implement Creating Functions**

- **Aim:** To demonstrate defining and using various types of functions
  in Python.

- **Algorithm/Approach:**

  1.  **Basic Function:** Define and call a simple function with one
      parameter.

  2.  **Return Value:** Define and call a function that returns a
      computed value.

  3.  **Default Parameter:** Define and call a function with a default
      parameter value, showing both default and overridden calls.

  4.  **Keyword Arguments:** Define and call a function using keyword
      arguments to specify parameters out of order.

  5.  **Variable Positional Arguments (\*args):** Define and call a
      function that accepts a variable number of positional arguments.

  6.  **Variable Keyword Arguments (\*\*kwargs):** Define and call a
      function that accepts arbitrary keyword arguments.

  7.  **Lambda Function:** Define and use a simple anonymous function.

- **Code/Program:**  
  \# --- Basic Function ---  
  def greet(name): print(f"Hello, {name}!")  
  greet("Demo User")  
    
  \# --- Function with Return Value ---  
  def add_numbers(x, y): return x + y  
  sum_result = add_numbers(15, 13)  
  print(f"\nSum: {sum_result}")  
    
  \# --- Function with Default Parameter Value ---  
  def power(base, exponent=2): return base \*\* exponent  
  print(f"\nPower (default): {power(5)}")  
  print(f"Power (override): {power(5, 3)}")  
    
  \# --- Function with Keyword Arguments ---  
  def describe_pet(animal_type, pet_name): print(f"\nPet: {animal_type}
  named {pet_name}.")  
  describe_pet(pet_name="Buddy", animal_type="dog")  
    
  \# --- Function with \*args ---  
  def sum_all(\*numbers): return sum(numbers)  
  print(f"\nSum (\*args): {sum_all(1, 2, 3, 4, 5)}")  
    
  \# --- Function with \*\*kwargs ---  
  def build_profile(first, last, \*\*user_info):  
  profile = {'first': first, 'last': last}  
  profile.update(user_info)  
  return profile  
  user = build_profile('jane', 'doe', city='Anytown', country='USA')  
  print(f"\nProfile (\*\*kwargs): {user}")  
    
  \# --- Lambda Function ---  
  multiply = lambda x, y: x \* y  
  print(f"\nLambda Multiply: {multiply(7, 6)}")

- **Output (Sample):**  
  Hello, Demo User!  
    
  Sum: 28  
    
  Power (default): 25  
  Power (override): 125  
    
  Pet: dog named Buddy.  
    
  Sum (\*args): 15  
    
  Profile (\*\*kwargs): {'first': 'jane', 'last': 'doe', 'city':
  'Anytown', 'country': 'USA'}  
    
  Lambda Multiply: 42

**2. Python - Kth Smallest Element in a BST**

- **Aim:** To find the kth smallest value (1-indexed) in a Binary Search
  Tree (BST).

- **Algorithm/Approach:** Iterative Inorder Traversal:

  1.  **Define Node:** Define TreeNode class.

  2.  **Initialize:** stack = \[\], count = 0, current = root.

  3.  **Traversal Loop:** While current or stack:

      - **Go Left:** While current: Push current to stack, current =
        current.left.

      - **Process Node:** Pop from stack to current. Increment count.

      - **Check K:** If count == k, return current.val.

      - **Go Right:** current = current.right.

  4.  **Handle Invalid K:** Return -1 if loop finishes.

- **Code/Program:**  
  from typing import Optional  
    
  class TreeNode:  
  def \_\_init\_\_(self, val=0, left=None, right=None):  
  self.val = val; self.left = left; self.right = right  
    
  def kthSmallest(root: Optional\[TreeNode\], k: int) -\> int:  
  stack = \[\]  
  count = 0  
  current = root  
  while current or stack:  
  while current:  
  stack.append(current)  
  current = current.left  
  current = stack.pop()  
  count += 1  
  if count == k:  
  return current.val  
  current = current.right  
  return -1  
    
  \# Example 1: root = \[3,1,4,null,2\], k = 1  
  root1 = TreeNode(3, TreeNode(1, None, TreeNode(2)), TreeNode(4))  
  k1 = 1  
  print("Example 1:")  
  print(f"Input: k={k1}")  
  print(f"Output: {kthSmallest(root1, k1)}")  
    
  \# Example 2: root = \[5,3,6,2,4,null,null,1\], k = 3  
  root2 = TreeNode(5, TreeNode(3, TreeNode(2, TreeNode(1)),
  TreeNode(4)), TreeNode(6))  
  k2 = 3  
  print("\nExample 2:")  
  print(f"Input: k={k2}")  
  print(f"Output: {kthSmallest(root2, k2)}")

- **Output (Sample):**  
  Example 1:  
  Input: k=1  
  Output: 1  
    
  Example 2:  
  Input: k=3  
  Output: 3

## **Experiment 8**

**1. Python - Implement Lists and Tuples**

- **Aim:** To demonstrate the creation, characteristics
  (mutability/immutability), and common operations for Python lists and
  tuples.

- **Algorithm/Approach:**

  1.  **Lists:** Create (\[\]), access elements (index/slice), modify
      elements, append (.append), insert (.insert), remove (.remove,
      .pop), get length (len), sort (.sort), reverse (.reverse). Print
      results at each stage.

  2.  **Tuples:** Create (()), access elements (index/slice),
      demonstrate immutability (attempt modification in try-except), get
      length (len), concatenate (+ creates new tuple). Print results.

- **Code/Program:**  
  \# --- Lists (Mutable) ---  
  print("--- Lists ---")  
  my_list = \[10, "apple", 25.5\]  
  print(f"Original: {my_list}")  
  my_list\[1\] = "banana" \# Modify  
  my_list.append(True) \# Append  
  print(f"Modified: {my_list}")  
  my_list.insert(0, "start") \# Insert  
  print(f"Inserted: {my_list}")  
  my_list.pop(2) \# Pop by index  
  print(f"Popped: {my_list}")  
  print(f"Length: {len(my_list)}")  
    
  \# --- Tuples (Immutable) ---  
  print("\n--- Tuples ---")  
  my_tuple = (10, "apple", 25.5)  
  print(f"Original: {my_tuple}")  
  print(f"Element 0: {my_tuple\[0\]}")  
  try: my_tuple\[0\] = 5  
  except TypeError as e: print(f"Modification attempt error: {e}")  
  new_tuple = my_tuple + (False,)  
  print(f"Concatenated: {new_tuple}")  
  print(f"Length: {len(my_tuple)}")

- **Output (Sample):**  
  --- Lists ---  
  Original: \[10, 'apple', 25.5\]  
  Modified: \[10, 'banana', 25.5, True\]  
  Inserted: \['start', 10, 'banana', 25.5, True\]  
  Popped: \['start', 10, 25.5, True\]  
  Length: 4  
    
  --- Tuples ---  
  Original: (10, 'apple', 25.5)  
  Element 0: 10  
  Modification attempt error: 'tuple' object does not support item
  assignment  
  Concatenated: (10, 'apple', 25.5, False)  
  Length: 3

**2. Python - Word Break**

- **Aim:** To determine if a string s can be segmented into a sequence
  of words from a dictionary wordDict.

- **Algorithm/Approach:** Dynamic Programming:

  1.  **Prepare:** Convert wordDict to a set (word_set). Get length n
      of s. Create dp = \[False\] \* (n + 1).

  2.  **Base Case:** dp\[0\] = True.

  3.  **Outer Loop:** Iterate i from 1 to n.

  4.  **Inner Loop:** Iterate j from 0 to i-1.

  5.  **Check:** If dp\[j\] is True and s\[j:i\] is in word_set:

      - Set dp\[i\] = True.

      - break inner loop.

  6.  **Return:** Return dp\[n\].

- **Code/Program:**  
  from typing import List  
    
  def wordBreak(s: str, wordDict: List\[str\]) -\> bool:  
  word_set = set(wordDict)  
  n = len(s)  
  dp = \[False\] \* (n + 1)  
  dp\[0\] = True  
  for i in range(1, n + 1):  
  for j in range(i):  
  if dp\[j\] and s\[j:i\] in word_set:  
  dp\[i\] = True  
  break  
  return dp\[n\]  
    
  \# Example usage:  
  s1 = "leetcode"; wordDict1 = \["leet", "code"\]  
  print(f"Input: s=\\{s1}\\, dict={wordDict1}")  
  print(f"Output: {wordBreak(s1, wordDict1)}")  
    
  s2 = "applepenapple"; wordDict2 = \["apple", "pen"\]  
  print(f"\nInput: s=\\{s2}\\, dict={wordDict2}")  
  print(f"Output: {wordBreak(s2, wordDict2)}")  
    
  s3 = "catsandog"; wordDict3 = \["cats", "dog", "sand", "and",
  "cat"\]  
  print(f"\nInput: s=\\{s3}\\, dict={wordDict3}")  
  print(f"Output: {wordBreak(s3, wordDict3)}")

- **Output (Sample):**  
  Input: s="leetcode", dict=\['leet', 'code'\]  
  Output: True  
    
  Input: s="applepenapple", dict=\['apple', 'pen'\]  
  Output: True  
    
  Input: s="catsandog", dict=\['cats', 'dog', 'sand', 'and', 'cat'\]  
  Output: False


> **LIST OF EXPERIMENTS**

<table>
<colgroup>
<col style="width: 6%" />
<col style="width: 9%" />
<col style="width: 17%" />
<col style="width: 8%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 12%" />
<col style="width: 9%" />
<col style="width: 13%" />
</colgroup>
<tbody>
<tr>
<td rowspan="2" style="text-align: center;"><blockquote>
<p><strong>S.No</strong></p>
</blockquote></td>
<td rowspan="2"><blockquote>
<p><strong>Date</strong></p>
</blockquote></td>
<td rowspan="2"><strong>Experiment Name</strong></td>
<td rowspan="2"><blockquote>
<p><strong>Page</strong></p>
<p><strong>No</strong></p>
</blockquote></td>
<td colspan="2"
style="text-align: center;"><strong>Performance</strong></td>
<td rowspan="2"><blockquote>
<p><strong>Viva Voice</strong></p>
</blockquote></td>
<td rowspan="2" style="text-align: center;"><strong>Total
Marks</strong></td>
<td rowspan="2" style="text-align: center;"><strong>Sign of the
faculty</strong></td>
</tr>
<tr>
<td style="text-align: center;"><strong>Lab( )</strong></td>
<td style="text-align: center;"><strong>Record( )</strong></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>1.</strong></p>
</blockquote></td>
<td></td>
<td><blockquote>
<p>Basic Python Programming</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>2.</strong></p>
</blockquote></td>
<td></td>
<td><blockquote>
<p>Use matrix transformations and linear algebra to process the data
using R.</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>3.</strong></p>
</blockquote></td>
<td></td>
<td><blockquote>
<p>Randomly split a large dataset into training and test sets using
R.</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>4.</strong></p>
</blockquote></td>
<td></td>
<td><blockquote>
<p>Apply various statistical and machine learning functions Using R.</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>5.</strong></p>
</blockquote></td>
<td></td>
<td><blockquote>
<p>Provide an analysis of the reliability</p>
<p>and goodness of fit of the results Using R.</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>6.</strong></p>
</blockquote></td>
<td></td>
<td><blockquote>
<p>Resume Parser Using AI</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><blockquote>
<p><strong>7.</strong></p>
</blockquote></td>
<td></td>
<td>Fake news detector using AI</td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>8.</strong></td>
<td></td>
<td><blockquote>
<p>Instagram Spam Detector using AI</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
<tr>
<td style="text-align: center;"><strong>9.</strong></td>
<td></td>
<td><blockquote>
<p>Animal Species Prediction using AI.</p>
</blockquote></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
<td></td>
<td style="text-align: center;"></td>
<td style="text-align: center;"></td>
</tr>
</tbody>
</table>

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 a)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
<p><strong>FizzBuzz</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

**AIM:**

To write a Python program that prints numbers from 1 to 100 with special
rules:

- Print "Fizz" for multiples of 3.

- Print "Buzz" for multiples of 5.

- Print "FizzBuzz" for multiples of both 3 and 5.

**ALGORITHM**:

1\. Start a loop from 1 to 100.

2\. For each number:

- If the number is divisible by both 3 and 5, print "FizzBuzz".

- Else if the number is divisible by 3, print "Fizz".

- Else if the number is divisible by 5, print "Buzz".

- Else, print the number itself.

3\. End the loop.

**PROGRAM:**

> for num in range(1, 101):
>
> if num % 3 == 0 and num % 5 == 0:
>
> print("FizzBuzz")
>
> elif num % 3 == 0:
>
> print("Fizz")
>
> elif num % 5 == 0:
>
> print("Buzz")
>
> else:
>
> print(num)

**OUTPUT:**

> 1
>
> 2
>
> Fizz
>
> 4
>
> Buzz
>
> Fizz
>
> 7
>
> 8
>
> Fizz
>
> Buzz
>
> 11
>
> Fizz
>
> 13
>
> 14
>
> FizzBuzz
>
> 16
>
> 17
>
> Fizz
>
> 19
>
> Buzz
>
> Fizz
>
> 22
>
> 23
>
> Fizz
>
> Buzz
>
> 26
>
> Fizz
>
> 28
>
> 29
>
> FizzBuzz

**RESULT:**

Thus the above program is executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 b)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>Basic Calculator</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To write a Python program that performs addition, subtraction,
> multiplication, and
>
> division based on user input.

###  ALGORITHM:

1.  Display options to the user (Add, Subtract, Multiply, Divide).

2.  Ask the user to select an operation.

3.  Ask the user to input two numbers.

4.  Based on the user's choice:

    - Perform addition if choice is 1.

    - Perform subtraction if choice is 2.

    - Perform multiplication if choice is 3.

    - Perform division if choice is 4.

5.  Display the result.

6.  End the program.

**PROGRAM:**

> print("Select operation:")
>
> print("1. Addition")
>
> print("2. Subtraction")
>
> print("3. Multiplication")
>
> print("4. Division")
>
> choice = input("Enter choice (1/2/3/4): ")
>
> \# Take input for two numbers
>
> num1 = float(input("Enter first number: "))
>
> num2 = float(input("Enter second number: "))
>
> \# Perform calculation based on choice
>
> if choice == '1':
>
> print("Result:", num1 + num2)
>
> elif choice == '2':
>
> print("Result:", num1 - num2)
>
> elif choice == '3':
>
> print("Result:", num1 \* num2)
>
> elif choice == '4':
>
> if num2 != 0:
>
> print("Result:", num1 / num2)
>
> else:
>
> print("Error: Division by zero is not allowed.")
>
> else:
>
> print("Invalid Input")
>
> **OUTPUT:**
>
> Select operation:
>
> 1\. Addition
>
> 2\. Subtraction
>
> 3\. Multiplication
>
> 4\. Division
>
> Enter choice (1/2/3/4): 4
>
> Enter first number: 10
>
> Enter second number: 0
>
> Error: Division by zero is not allowed.

**RESULT:**

Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 c)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>Count Vowels</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To write a Python function that counts the number of vowels (a, e, i,
> o, u) in a given string.

### ALGORITHM:

1.  Define a function that takes a string as input.

2.  Initialize a counter to 0.

3.  For each character in the string:

    - Check if it is a vowel (either lowercase or uppercase).

    - If yes, increment the counter.

4.  Return or print the counter value.

**PROGRAM:**

> def count_vowels(string):
>
> vowels = 'aeiouAEIOU'
>
> count = 0
>
> for char in string:
>
> if char in vowels:
>
> count += 1
>
> return count
>
> \# Take input from the user
>
> text = input("Enter a string: ")
>
> \# Call the function and display the result
>
> print("Number of vowels:", count_vowels(text))

**OUTPUT:**

Enter a string: Hello World

> Number of vowels: 3

**RESULT:**

Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 d)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>Temperature Converter</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To write a Python program that converts temperature from Celsius to
> Fahrenheit
>
> and Fahrenheit to Celsius based on user choice.

### ALGORITHM

1.  Display options to the user:

    - Convert Celsius to Fahrenheit

    - Convert Fahrenheit to Celsius

2.  Ask the user to select an option.

3.  Ask the user to enter the temperature value.

4.  If the user selects:

    - Celsius to Fahrenheit:  
      Use formula: F = (C × 9/5) + 32

    - Fahrenheit to Celsius:  
      Use formula: C = (F - 32) × 5/9

5.  Display the converted temperature.

**PROGRAM:**

> print("Temperature Converter")
>
> print("1. Celsius to Fahrenheit")
>
> print("2. Fahrenheit to Celsius")
>
> choice = input("Enter your choice (1/2): ")
>
> if choice == '1':
>
> celsius = float(input("Enter temperature in Celsius: "))
>
> fahrenheit = (celsius \* 9/5) + 32
>
> print(f"{celsius}°C is equal to {fahrenheit}°F")
>
> elif choice == '2':
>
> fahrenheit = float(input("Enter temperature in Fahrenheit: "))
>
> celsius = (fahrenheit - 32) \* 5/9
>
> print(f"{fahrenheit}°F is equal to {celsius}°C")
>
> else:
>
> print("Invalid choice!")

**OUTPUT:**

> **Temperature Converter**
>
> 1\. Celsius to Fahrenheit
>
> 2\. Fahrenheit to Celsius
>
> Enter your choice (1/2): 1
>
> Enter temperature in Celsius: 37
>
> 37.0°C is equal to 98.6°F
>
> **Temperature Converter**
>
> 1\. Celsius to Fahrenheit
>
> 2\. Fahrenheit to Celsius
>
> Enter your choice (1/2): 2
>
> Enter temperature in Fahrenheit: 212
>
> 212.0°F is equal to 100.0°C

**RESULT:**

Thus the above program was verified executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 e)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>Anagram Check</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To write a Python function that checks if two given strings are
> **anagrams** (i.e., they
>
> contain the same characters in a different order).

### ALGORITHM:

1.  Define a function that takes two strings as input.

2.  Remove any spaces and convert both strings to lowercase.

3.  Sort the characters of both strings.

4.  Compare the sorted versions:

    - If they are the same, the strings are anagrams.

    - Otherwise, they are not anagrams.

> **PROGRAM:**
>
> def is_anagram(str1, str2):
>
> \# Remove spaces and convert to lowercase
>
> str1 = str1.replace(" ", "").lower()
>
> str2 = str2.replace(" ", "").lower()
>
> \# Sort and compare
>
> if sorted(str1) == sorted(str2):
>
> return True
>
> else:
>
> return False
>
> \# Take input from user
>
> string1 = input("Enter first string: ")
>
> string2 = input("Enter second string: ")
>
> \# Call the function and display result
>
> if is_anagram(string1, string2):
>
> print("The strings are anagrams.")
>
> else:
>
> print("The strings are not anagrams.")

**OUTPUT:**

Enter first string: listen

> Enter second string: silent
>
> The strings are anagrams.
>
> Enter first string: hello
>
> Enter second string: world
>
> The strings are not anagrams.

**RESULT:**

Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 f)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>Raindrop Sounds Converter</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To write a Python program that converts a given number into a string
> containing specific "raindrop sounds" based on these rules:

- If the number is divisible by **3**, add "Pling" to the result.

- If the number is divisible by **5**, add "Plang" to the result.

- If the number is divisible by **7**, add "Plong" to the result.

- If the number is not divisible by 3, 5, or 7, the result should be the
  number itself as a string.

### ALGORITHM:

1.  Define a function that takes a number as input.

2.  Initialize an empty string for the result.

3.  Check divisibility:

    - If divisible by 3, add "Pling" to the result.

    - If divisible by 5, add "Plang" to the result.

    - If divisible by 7, add "Plong" to the result.

4.  If the result string is still empty, convert the number to a string.

5.  Return or print the result.

**PROGRAM:**

> def raindrop_sounds(number):
>
> result = ""
>
> if number % 3 == 0:
>
> result += "Pling"
>
> if number % 5 == 0:
>
> result += "Plang"
>
> if number % 7 == 0:
>
> result += "Plong"
>
> if result == "":
>
> result = str(number)
>
> return result
>
> \# Take input from user
>
> num = int(input("Enter a number: "))
>
> \# Display the result
>
> print("Raindrop sound:", raindrop_sounds(num))

**OUTPUT:**

Enter a number: 28

Raindrop sound: Plong

Enter a number: 30

> Raindrop sound: PlingPlang
>
> Enter a number: 34
>
> Raindrop sound: 34
>
> **RESULT:**
>
> Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 g)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>School Paper Editor</strong></p></td>
</tr>
</tbody>
</table>

### AIM:

> To write a Python program that provides four functions to edit a
> school paper by
>
> capitalizing titles, checking sentence endings, cleaning spaces, and
> replacing words.

### ALGORITHM:

1.  **capitalize_title(title)**:

    - Use string methods to capitalize the first letter of each word.

2.  **check_sentence_ending(sentence)**:

    - Check if the last character of the sentence is a period ..

3.  **clean_up_spacing(sentence)**:

    - Remove any spaces at the beginning and end of the sentence using
      strip().

4.  **replace_word_choice(sentence, old_word, new_word)**:

    - Replace all instances of a word using replace().

**PROGRAM:**

> def capitalize_title(title):
>
> \# Capitalize the first letter of each word
>
> return title.title()
>
> def check_sentence_ending(sentence):
>
> \# Check if the sentence ends with a period
>
> return sentence.endswith('.')
>
> def clean_up_spacing(sentence):
>
> \# Remove leading and trailing spaces
>
> return sentence.strip()
>
> def replace_word_choice(sentence, old_word, new_word):
>
> \# Replace old_word with new_word
>
> return sentence.replace(old_word, new_word)
>
> \# Example usage:
>
> title = input("Enter the title: ")
>
> sentence = input("Enter the sentence: ")
>
> old_word = input("Enter the word you want to replace: ")
>
> new_word = input("Enter the new word: ")
>
> print("\nEdited Title:", capitalize_title(title))
>
> print("Does the sentence end with a period?",
> check_sentence_ending(sentence))
>
> print("Cleaned-up sentence:", clean_up_spacing(sentence))
>
> print("Updated sentence:", replace_word_choice(sentence, old_word,
> new_word))
>
> **OUTPUT:**
>
> Enter the title: the life of a student
>
> Enter the sentence: This is an example sentence
>
> Enter the word you want to replace: example
>
> Enter the new word: sample
>
> Edited Title: The Life Of A Student
>
> Does the sentence end with a period? False
>
> Cleaned-up sentence: This is an example sentence
>
> Updated sentence: This is an sample sentence
>
> **RESULT:**
>
> Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 h)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>Ugly Number Checker</strong></p></td>
</tr>
</tbody>
</table>

### AIM:

> To implement a Python program that determines whether a given number
> is an
>
> **Ugly Number** (only prime factors allowed: 2, 3, and 5).

### ALGORITHM:

1.  Define a function that takes a number as input.

2.  If the number is 1, return True.

3.  If the number is negative or zero, return False.

4.  While the number is divisible by 2, divide it by 2.

5.  While the number is divisible by 3, divide it by 3.

6.  While the number is divisible by 5, divide it by 5.

7.  After all divisions, if the number becomes 1, it is an Ugly Number
    (True); else, it is not (False).

> **PROGRAM:**
>
> def is_ugly(number):
>
> if number == 1:
>
> return True
>
> if number \<= 0:
>
> return False
>
> for factor in \[2, 3, 5\]:
>
> while number % factor == 0:
>
> number = number // factor
>
> return number == 1
>
> \# Take input from the user
>
> num = int(input("Enter a number: "))
>
> \# Check and display result
>
> if is_ugly(num):
>
> print(num, "is an Ugly Number.")
>
> else:
>
> print(num, "is NOT an Ugly Number.")

**OUTPUT:**

Enter a number: 6

> 6 is an Ugly Number.
>
> Enter a number: 14
>
> 14 is NOT an Ugly Number.
>
> Enter a number: -5
>
> -5 is NOT an Ugly Number.
>
> **RESULT:**
>
> Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 i)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Basic Python Programming</strong></p>
</blockquote>
<p><strong>FizzBuzz with different string formatting
methods</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To implement a Python function that generates the FizzBuzz sequence
> from 1 to n, and formats the output using:

- % formatting

- .format() method

- f-strings**  
  **

### ALGORITHM:

1.  Define a function that takes a number n as input.

2.  Loop from 1 to n (inclusive).

3.  For each number:

    - If divisible by 3 and 5, output "FizzBuzz".

    - Else if divisible by 3, output "Fizz".

    - Else if divisible by 5, output "Buzz".

    - Else, output the number itself as a string.

4.  Format and display the output using:

    - % formatting**  
      **

    - .format()

**PROGRAM:**

> def fizzbuzz_formatting(n):
>
> print("\nUsing % formatting:")
>
> for i in range(1, n + 1):
>
> if i % 3 == 0 and i % 5 == 0:
>
> print("%s" % "FizzBuzz")
>
> elif i % 3 == 0:
>
> print("%s" % "Fizz")
>
> elif i % 5 == 0:
>
> print("%s" % "Buzz")
>
> else:
>
> print("%s" % str(i))
>
> print("\nUsing .format() method:")
>
> for i in range(1, n + 1):
>
> if i % 3 == 0 and i % 5 == 0:
>
> print("{}".format("FizzBuzz"))
>
> elif i % 3 == 0:
>
> print("{}".format("Fizz"))
>
> elif i % 5 == 0:
>
> print("{}".format("Buzz"))
>
> else:
>
> print("{}".format(i))
>
> print("\nUsing f-strings:")
>
> for i in range(1, n + 1):
>
> if i % 3 == 0 and i % 5 == 0:
>
> print(f"{'FizzBuzz'}")
>
> elif i % 3 == 0:
>
> print(f"{'Fizz'}")
>
> elif i % 5 == 0:
>
> print(f"{'Buzz'}")
>
> else:
>
> print(f"{i}")
>
> \# Take input from user
>
> num = int(input("Enter the value of n: "))
>
> \# Call the function
>
> fizzbuzz_formatting(num)

**OUTPUT:**

Enter the value of n: 5

> Using % formatting:
>
> 1
>
> 2
>
> Fizz
>
> 4
>
> Buzz
>
> Using .format() method:
>
> 1
>
> 2
>
> Fizz
>
> 4
>
> Buzz
>
> Using f-strings:
>
> 1
>
> 2
>
> Fizz
>
> 4
>
> Buzz

**RESULT:**

Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 1 j)</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>BASIC PYTHON PROGRAMMING</strong></p>
</blockquote>
<p><strong>FRAUDULENT TRANSACTION DETECTION</strong></p></td>
</tr>
</tbody>
</table>

**AIM:**

> To implement a Python program that detects fraudulent transactions
> based on

statistical measures like mean and standard deviation.  
A transaction is considered fraudulent if it exceeds (Mean + 2 × Std
Dev).

### ALGORITHM:

1.  Define a list of transaction amounts.

2.  Calculate the mean (average) of the transactions.

3.  Calculate the standard deviation of the transactions.

4.  Set the fraud detection threshold as Mean + 2 × Standard Deviation.

5.  Check each transaction:

    - If it is greater than the threshold, mark it as fraudulent.

6.  Display all fraudulent transactions.

**PROGRAM:**

> def calculate_mean(data):
>
> return sum(data) / len(data)
>
> \# Function to calculate standard deviation
>
> def calculate_std(data, mean):
>
> variance = sum((x - mean) \*\* 2 for x in data) / len(data)
>
> return variance \*\* 0.5
>
> \# Function to detect fraudulent transactions
>
> def detect_fraud(transactions):
>
> mean = calculate_mean(transactions)
>
> std = calculate_std(transactions, mean)
>
> threshold = mean + 2 \* std
>
> print(f"Mean of transactions: {mean}")
>
> print(f"Standard Deviation of transactions: {std}")
>
> print(f"Fraud Threshold: {threshold}\n")
>
> print("Fraudulent Transactions:")
>
> for amount in transactions:
>
> if amount \> threshold:
>
> print(amount)
>
> \# Example transaction data
>
> transactions = \[100, 120, 130, 150, 110, 105, 500, 115, 140, 108\]
>
> \# Call the function
>
> detect_fraud(transactions)

**OUTPUT:**

Mean of transactions: 157.8

> Standard Deviation of transactions: 110.13224581594603
>
> Fraud Threshold: 378.06449163189205
>
> Fraudulent Transactions:
>
> 500

**RESULT:**

Thus the above program was executed successfully.

<table>
<colgroup>
<col style="width: 21%" />
<col style="width: 78%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 2</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Use Matrix Transformation and Linear Algebra to process the
data using R</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# AIM:

> To process data using matrix transformations and linear algebra
> techniques in R.

# ALGORITHM:

1.  Prints the original data matrix.

2.  Centers the data by subtracting the mean of each column.

3.  Prints the centered data matrix.

4.  Scales the data by scaling each column to have mean 0 and standard
    deviation 1.

5.  Prints the scaled data matrix.

6.  Performs matrix multiplication by multiplying the data matrix by its
    transpose.

7.  Prints the result of the matrix multiplication.

# PROGRAM:

> \# Sample data (5 data points with 2 features) data \<- matrix(c(1, 2,
> 3, 4, 5, 6, 7, 8, 9, 10), ncol = 2, byrow = TRUE) print("Original
> Data:") print(data)
>
> \# Centering the data (subtracting mean of each column) centered_data
> \<- scale(data, center = TRUE, scale = FALSE) print("Centered Data:")
>
> print(centered_data)
>
> \# Scaling the data (scaling each column to have mean 0 and standard
> deviation 1)
>
> scaled_data \<- scale(data) print("Scaled Data:") print(scaled_data)
>
> \# Performing matrix multiplication
>
> \# For example, let's multiply the data matrix by its transpose
> matrix_multiplication \<- data %\*% t(data)
>
> print("Matrix Multiplication:") print(matrix_multiplication)

#  OUTPUT:

<img src="media/image1.jpg" style="width:1.94992in;height:5.59271in" />

> **RESULT:**
>
> Thus the above program is executed successfully.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 3 Date:</p>
</blockquote></td>
<td><blockquote>
<p><strong>Randomly split a large dataset into training and test sets
using R</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# AIM:

To Randomly split a large dataset into training and test sets using R.

# ALGORITHM:

1.  Loads the 'mtcars' dataset.

2.  Assigns the dataset to a variable named my_data.

3.  Sets a seed for reproducibility using set.seed(123).

4.  Generates random indices for the training and test sets. 80% of the
    data are randomly selected for the training set, and the remaining
    are used for the test set.

5.  Creates the training set (train_data) by subsetting my_data based on
    the generated training indices.

6.  Creates the test set (test_data) by subsetting my_data based on the
    generated test indices.

7.  Prints the dimensions of the training and test sets

# PROGRAM:

> data(mtcars) \# Load the 'mtcars' dataset my\_
>
> data\<- mtcars
>
> my_data \<- iris \# Load the 'iris' dataset from the datasets package
>
> set.seed(123)
>
> \# Generate random indices for training and test sets n \<-
> nrow(my_data) \# Number of rows in the dataset
>
> train_indices \<- sample(1:n, size = 0.8\*n, replace = FALSE) \# 80%
> for training test_indices \<- setdiff(1:n, train_indices) \# Remaining
> for testing
>
> \# Create training and test sets using the generated indices
>
> train_data\<- my_data\[train_indices, \]
>
> test_data\<-my_data\[test_indices, \]
>
> \# Print dimensions of training and test sets cat("Training set
> dimensions:", dim(train_data), "\n")
>
> cat("Test set dimensions:", dim(test_data), "\n"
>
> **OUTPUT:**
>
> Training set dimensions: 120 5
>
> Test set dimensions: 30 5
>
> **RESULT :**
>
> Thus the above program is executed successfully

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 4</p>
<p>Date:</p>
</blockquote></td>
<td><blockquote>
<p><strong>Apply various statistical and machine learning functions
Using R</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# 

# AIM:

> To Apply various statistical and machine learning functions Using R.

# ALGORITHM:

1.  Loads the mtcars dataset.

2.  Checks the structure of the dataset using str() to understand its
    variables.

3.  Fits a linear regression model to predict mpg (miles per gallon)
    using cyl (number of cylinders) as the predictor variable.

4.  Summarizes the fitted regression model using summary().

5.  Makes predictions on the mtcars dataset using the fitted model.

6.  Prints the first few predicted values.

# PROGRAM:

> \# Load the built-in mtcars dataset
>
> data(mtcars)
>
> \# Check the structure of the dataset
>
> str(mtcars)
>
> \# Fit a linear regression model to predict mpg using the number of
> cylinders
>
> model \<- lm(mpg ~ cyl, data = mtcars)
>
> \# Summary of the regression model
>
> summary(model)
>
> \# Make predictions using the fitted model
>
> predictions \<- predict(model, newdata = mtcars)
>
> \# Print the first few predicted values
>
> cat("First few predicted mpg values:\n")
>
> head(predictions)
>
> \# Model evaluation: R-squared and RMSE
>
> r_squared \<- summary(model)\$r.squared
>
> rmse \<- sqrt(mean((mtcars\$mpg - predictions)^2))
>
> cat("\nR-squared:", r_squared, "\n")
>
> cat("Root Mean Squared Error (RMSE):", rmse, "\n")
>
> \# Visualization of the regression line
>
> plot(mtcars\$cyl, mtcars\$mpg,
>
> main = "Linear Regression: MPG vs Cylinders",
>
> xlab = "Number of Cylinders", ylab = "Miles Per Gallon (mpg)",
>
> pch = 19, col = "blue")
>
> \# Add regression line
>
> abline(model, col = "red", lwd = 2)
>
> \# Add predicted values as points
>
> points(mtcars\$cyl, predictions, col = "green", pch = 4)
>
> legend("topright", legend = c("Actual", "Predicted", "Regression
> Line"),
>
> col = c("blue", "green", "red"), pch = c(19, 4, NA), lty = c(NA, NA,
> 1),
>
> lwd = c(NA, NA, 2))

# OUTPUT:

<img src="media/image2.jpg" style="width:6.46889in;height:2.02083in" />

> **RESULT :**
>
> Thus the above program is executed successfully.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 5 Date:</p>
</blockquote></td>
<td><blockquote>
<p><strong>Provide an analysis of the reliability and goodness of fit of
the results Using R</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# AIM:

> To Provide an analysis of the reliability and goodness of fit of the
> results Using R

**ALGORITHM:**

1.  Load the 'mtcars' dataset.

2.  Create training and test sets by randomly splitting the dataset.

3.  Fit a linear regression model using the training set.

4.  Make predictions on both the training and test sets.

5.  Compute Mean Squared Error (MSE) and R-squared for both sets.

6.  Create visualizations to analyze the distribution of variables and
    relationships between them.

# PROGRAM:

> \# Load the 'mtcars' dataset data(mtcars) my_data \<- mtcars
>
> \# Set seed for reproducibility set.seed(123)
>
> \# Generate random indices for training and test sets n \<-
> nrow(my_data) \# Number of rows in the dataset
>
> train_indices \<- sample(1:n, size = 0.8\*n, replace = FALSE) \# 80%
> for training
>
> test_indices \<- setdiff(1:n, train_indices) \# Remaining for testing
>
> \# Create training and test sets using the generated indices
> train_data
>
> \<- my_data\[train_indices, \] test_data \<- my_data\[test_indices, \]
>
> \# Fit a linear regression model using the training set model
>
> \<- lm(mpg ~ ., data = train_data)
>
> \# Make predictions on both training and test sets train_predictions
> \<- predict(model, newdata = train_data) test_predictions \<-
> predict(model, newdata = test_data)
>
> \# Compute Mean Squared Error (MSE)
>
> train_mse \<- mean((train_data\$mpg - train_predictions)^2)
>
> test_mse\<- mean((test_data\$mpg - test_predictions)^2)
>
> \# Compute R-squared
>
> train_r_squared \<- summary(model)\$r.squared
>
> test_r_squared\<- cor(test_data\$mpg, test_predictions)^2
>
> \# Print the results cat("Training Set Metrics:\n") cat("Mean Squared
> Error (MSE):", train_mse, "\n") cat("R-squared:", train_r_squared,
> "\n\n")
>
> cat("Test Set Metrics:\n")
>
> cat("Mean Squared Error (MSE):", test_mse, "\n") cat("R-squared:",
> test_r_squared, "\n")
>
> \# Load required libraries library(ggplot2)
>
> \# Histogram of mpg (miles per gallon) ggplot(my_data, aes(x = mpg)) +
> geom_histogram(binwidth = 2, fill = "skyblue", color = "black") +
> labs(title = "Distribution of Miles Per Gallon", x = "Miles Per
> Gallon", y = "Frequency")
>
> \# Scatter plot of mpg vs. horsepower (hp) ggplot(my_data, aes(x = hp,
> y = mpg)) + geom_point(color = "blue")
>
> labs(title = "Scatter Plot of Miles Per Gallon vs. Horsepower", x =
> "Horsepower",
>
> y = "Miles Per Gallon")
>
> \# Scatter plot of mpg vs. weight (wt)
>
> ggplot(my_data, aes(x= wt, y = mpg)) + geom_point(color = "green") +
> labs(title
>
> = "Scatter Plot of Miles Per Gallon vs. Weight", x = "Weight",y =
> "Miles Per Gallon")
>
> \# Scatter plot of mpg vs. number of cylinders (cyl) ggplot(my_data,
> aes(x = cyl, y = mpg)) + geom_point(color = "red") + labs(title =
> "Scatter Plot of Miles Per Gallon vs. Number of Cylinders", x =
> "Number of Cylinders", y = "Miles Per

# OUTPUT:

> Test Set Metrics:
>
> Mean Squared Error (MSE): 19.13392
>
> R-squared: 0.2879197
>
> \$x
>
> \[1\] "Horsepower"
>
> \$y
>
> \[1\] "Miles Per Gallon"
>
> \$title
>
> \[1\] "Scatter Plot of Miles Per Gallon vs. Horsepower"
>
> attr(,"class")
>
> \[1\] "labels"

<img src="media/image3.png" style="width:4.51806in;height:3.75in" />

# RESULT :

> Thus the above program is executed successfully.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 6</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Resume Parser Using AI</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# AIM:

> To Build a Resume Parser Using AI

# ALGORITHM:

1.  Input:

2.  Receive a resume document as input.

3.  Text Extraction:

4.  Extract text from the resume document.

5.  Information Extraction:

6.  Use regular expressions to extract basic information such as name,
    email, phone number, education, and work experience.

7.  Print or store the extracted information.

# PROGRAM:

> import re
>
> \# Sample resume text resume_text
>
> = """
>
> John Doe
>
> 123 Main St, Anytown, USA <john.doe@email.com>
>
> \(123\) 456-7890
>
> Education:
>
> Bachelor of Science in Computer Science University of Anytown,
> Anytown, USA
>
> Graduated: May 2015
>
> Work Experience:
>
> Software Engineer
>
> XYZ Tech, Anytown, USA
>
> June 2015 - Present

- Developed web applications using Django framework

- Implemented RESTful APIs for data exchange """

> \# Define regular expressions for extracting information name_regex =
> re.compile(r'^(\[A-Z\]\[a-z\]+)\s(\[A-Z\]\[a-z\]+)\$') email_regex =
>
> re.compile(r'\b\[A-Za-z0-9.\_%+-\]+@\[A-Za-z0-9.-\]+\\\[A-Z\|a-
> z\]{2,}\b')
>
> phone_regex = re.compile(r'\\?\d{3}\\?\[-.\s\]?\d{3}\[-.\s\]?\d{4}')
>
> \# Extract name
>
> match_name = name_regex.search(resume_text)
>
> if match_name:
>
> full_name = match_name.group() print("Name:", full_name)
>
> \# Extract email
>
> match_email = email_regex.search(resume_text)
>
> if match_email:
>
> email = match_email.group() print("Email:", email)
>
> \# Extract phone number
>
> match_phone = phone_regex.search(resume_text)
>
> if match_phone:
>
> phone_number = match_phone.group() print("Phone Number:",
> phone_number)
>
> \# Extract education
>
> education_match=re.search(r'Education:(.\*?)Work Experience:',
> resume_text, re.DOTALL)
>
> if education_match:
>
> education_details= education_match.group(1).strip()
>
> print("Education:")
>
> print(education_details)
>
> \# Extract work experience
>
> work_experience_match = re.search(r'Work Experience:(.\*?)\$',
> resume_text, re.DOTALL)
>
> ifwork_experience_match:
>
> work_experience_details=work_experience_match.group(1).strip()
> print("Work Experience:")
>
> print(work_experience_details)

# 

# OUTPUT:

<img src="media/image4.jpg" style="width:5.39566in;height:2.77083in" />

> **RESULT:**
>
> Thus the above program is executed successfully.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 7</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>Fake news detector using AI</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# AIM:

> To Build a Fake news detector using AI

#  ALGORITHM:

1.  Data Collection: Gather labeled news articles dataset.

2.  Preprocessing: Clean, tokenize, and normalize text data.

3.  Feature Extraction: Convert text into numerical features (TF-IDF,
    word embeddings).

4.  Model Training: Train a machine learning model (e.g., logistic
    regression).

5.  Evaluation: Assess model performance using accuracy metrics.

6.  Deployment: Deploy the trained model for real-time detection.

7.  Feedback Loop: Continuously improve the model based on feedback.

# PROGRAM:

> import pandas as pd
>
> from sklearn.model_selection
>
> import train_test_split
>
> from sklearn.feature_extraction.text
>
> import TfidfVectorizer
>
> from sklearn.linear_model
>
> import LogisticRegression
>
> from sklearn.metrics import classification_report, accuracy_score
>
> import nltk
>
> from nltk.corpus import stopwords
>
> nltk.download('stopwords')
>
> stop_words = set(stopwords.words('english'))
>
> data = {
>
>     'text': \[
>
>         'Climate change is a hoax.',
>
>         'The Earth revolves around the Sun.',
>
>         '5G technology spreads viruses.',
>
>         'Scientists discover a new species in the Amazon.',
>
>         'Eating carrots improves night vision.',
>
>         'Electric cars are better for the environment.'
>
>     \],
>
>     'label': \[1, 0, 1, 0, 1, 0\]
>
> }
>
> df = pd.DataFrame(data)
>
> df\['text'\] = df\['text'\].str.lower().str.replace('\[^\w\s\]', '',
> regex=True)
>
> df\['text'\] = df\['text'\].apply(lambda x: ' '.join(\[word for word
> in x.split() if word not in stop_words\]))
>
> X_train, X_test, y_train, y_test = train_test_split(df\['text'\],
> df\['label'\], test_size=0.3, random_state=42)
>
> tfidf = TfidfVectorizer()
>
> X_train_vec = tfidf.fit_transform(X_train)
>
> X_test_vec = tfidf.transform(X_test)
>
> model = LogisticRegression()
>
> model.fit(X_train_vec, y_train)
>
> y_pred = model.predict(X_test_vec)
>
> print("Accuracy:", accuracy_score(y_test, y_pred))
>
> print("ClassificationReport:\n", classification_report(y_test,
> y_pred))
>
> sample = \["Vaccines contain microchips"\]
>
> sample_vec = tfidf.transform(sample)
>
> prediction = model.predict(sample_vec)
>
> print(f"Prediction for sample: {'Fake' if prediction\[0\]==1 else
> 'Real'}")

# OUTPUT:

> Classification Report:

| Label        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.50      | 1.00   | 0.67     | 1       |
| 1            | 0.00      | 0.00   | 0.00     | 1       |
| Accuracy     |           |        | 0.50     | 2       |
| Macro Avg    | 0.25      | 0.50   | 0.33     | 2       |
| Weighted Avg | 0.25      | 0.50   | 0.33     | 2       |

> Prediction for sample: Real
>
> **RESULT :**
>
> Thus the above program is executed successfully.

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 8</p>
<p>Date:</p>
</blockquote></td>
<td><blockquote>
<p><strong>Instagram Spam Detector using AI</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# AIM:

> To Build a Instagram Spam Detector using AI

# ALGORITHM:

1.  Data Collection: Gather a dataset of Instagram comments labeled as
    spam or not spam.

2.  Preprocessing: Clean, tokenize, and normalize text data.

3.  Feature Extraction: Convert text into numerical features (TF-IDF,
    word embeddings).

4.  Model Training: Train a machine learning model (e.g., logistic
    regression).

5.  Evaluation: Assess model performance using accuracy metrics.

6.  Deployment: Integrate the trained model for real-time detection.

7.  Feedback Loop: Continuously improve the model based on feedback

# PROGRAM:

> import pandas as pd
>
> from sklearn.model_selection import train_test_split
>
> from sklearn.feature_extraction.text import TfidfVectorizer
>
> from sklearn.linear_model import LogisticRegression
>
> from sklearn.metrics import classification_report, accuracy_score
>
> import nltk
>
> from nltk.corpus import stopwords
>
> nltk.download('stopwords')
>
> stop_words = set(stopwords.words('english'))
>
> data = {
>
>     'message': \[
>
>        
>
> 'Get rich quick with this one simple trick!',
>
>         'Your profile is amazing, keep it up!',
>
>         'Claim your free gift card now!',
>
>         'Let’s work together on a project!',
>
>         'You’ve won a lottery! Click here to claim.',
>
>         'Great post! Looking forward to more.',
>
>         'Exclusive offer just for you! Act now!',
>
>         'Thanks for sharing such valuable content!'
>
>     \],
>
>     'label': \[1, 0, 1, 0, 1, 0, 1, 0\]
>
> }
>
> df = pd.DataFrame(data)
>
> df\['message'\] = df\['message'\].str.lower().str.replace('\[^\w\s\]',
> '', regex=True)
>
> df\['message'\] = df\['message'\].apply(lambda x: ' '.join(\[word for
> word in x.split() if word not in stop_words\]))
>
> X_train, X_test, y_train, y_test = train_test_split(df\['message'\],
> df\['label'\], test_size=0.3, random_state=42)
>
> tfidf = TfidfVectorizer()
>
> X_train_vec = tfidf.fit_transform(X_train)
>
> X_test_vec = tfidf.transform(X_test)
>
> model = LogisticRegression()
>
> model.fit(X_train_vec, y_train)
>
> y_pred = model.predict(X_test_vec)
>
> print("Accuracy:", accuracy_score(y_test, y_pred))
>
> print("Classification Report:\n", classification_report(y_test,
> y_pred))
>
> sample = \["Hey! You just won a free vacation. DM us now!"\]
>
> sample_vec = tfidf.transform(sample)
>
> prediction = model.predict(sample_vec)
>
> print(f"Prediction for sample: {'Spam' if prediction\[0\]==1 else 'Not
> Spam'}")

# 

# 

# 

# 

# 

# 

# OUTPUT:

Classification Report:

| Label        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.00      | 0.00   | 0.00     | 2       |
| 1            | 0.33      | 1.00   | 0.50     | 1       |
| Accuracy     |           |        | 0.33     | 3       |
| Macro Avg    | 0.17      | 0.50   | 0.25     | 3       |
| Weighted Avg | 0.11      | 0.33   | 0.17     | 3       |

Prediction for sample: Spam

> **RESULT**:
>
> Thus the above program is executed successfully

<table>
<colgroup>
<col style="width: 20%" />
<col style="width: 79%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Ex No: 9</p>
<p>Date:</p>
</blockquote></td>
<td style="text-align: center;"><blockquote>
<p><strong>ANIMAL SPECIES PREDICTION USING AI</strong></p>
</blockquote></td>
</tr>
</tbody>
</table>

# 

# AIM:

> To Build a Animal Species Prediction using AI.

# ALGORITHM:

> 1\. Collect a dataset with text descriptions and corresponding animal
> species
>
> labels.  
> 2.Clean and preprocess the text by lowering case, removing
> punctuation, and
>
> eliminating stopwords.  
> 3.Split the dataset into training and testing sets for evaluation.  
> 4.Convert text data into numerical form using TF-IDF vectorization.  
> 5.Train a Logistic Regression model on the TF-IDF features and species
> labels.  
> 6.Evaluate the model using accuracy and classification metrics on the
> test set.  
> 7.Use the trained model to predict species from new text descriptions.

# PROGRAM:

> import pandas as pd
>
> from sklearn.model_selection import train_test_split
>
> from sklearn.feature_extraction.text import TfidfVectorizer
>
> from sklearn.linear_model import LogisticRegression
>
> from sklearn.metrics import classification_report, accuracy_score
>
> import nltk
>
> from nltk.corpus import stopwords
>
> nltk.download('stopwords')
>
> stop_words = set(stopwords.words('english'))
>
> data = {
>
> 'description': \[
>
> 'Fast-running bird that cannot fly',
>
> 'Large mammal with antlers',
>
> 'Small amphibian that hops and croaks',
>
> 'Marine creature with eight legs',
>
> 'Striped horse-like animal found in Africa',
>
> 'Large bird of prey with sharp talons',
>
> 'Reptile with a hard shell',
>
> 'Small insect that produces honey'
>
> \],
>
> 'species': \[
>
> 'Ostrich', 'Deer', 'Frog', 'Octopus', 'Zebra', 'Eagle', 'Turtle',
> 'Bee'
>
> \]
>
> }
>
> df = pd.DataFrame(data)
>
> df\['description'\] =
> df\['description'\].str.lower().str.replace('\[^\w\s\]', '',
> regex=True)
>
> df\['description'\] = df\['description'\].apply(lambda x: '
> '.join(\[word for word in x.split() if word not in stop_words\]))
>
> X_train, X_test, y_train, y_test =
> train_test_split(df\['description'\], df\['species'\], test_size=0.3,
> random_state=42)
>
> tfidf = TfidfVectorizer()
>
> X_train_vec = tfidf.fit_transform(X_train)
>
> X_test_vec = tfidf.transform(X_test)
>
> model = LogisticRegression(max_iter=1000)
>
> model.fit(X_train_vec, y_train)
>
> y_pred = model.predict(X_test_vec)
>
> print("Accuracy:", accuracy_score(y_test, y_pred))
>
> print("Classification Report:\n", classification_report(y_test,
> y_pred))
>
> sample = \["Small insect that produces honey"\]
>
> sample_vec = tfidf.transform(sample)
>
> prediction = model.predict(sample_vec)
>
> print(f"Prediction for sample: {prediction\[0\]}")
>
> **OUTPUT:**
>
> Classification Report:

<table>
<colgroup>
<col style="width: 25%" />
<col style="width: 19%" />
<col style="width: 16%" />
<col style="width: 19%" />
<col style="width: 18%" />
</colgroup>
<thead>
<tr>
<th><blockquote>
<p>Label</p>
</blockquote></th>
<th><blockquote>
<p>Precision</p>
</blockquote></th>
<th><blockquote>
<p>Recall</p>
</blockquote></th>
<th><blockquote>
<p>F1-Score</p>
</blockquote></th>
<th><blockquote>
<p>Support</p>
</blockquote></th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p>Deer</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>1.0</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>Eagle</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>1.0</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>Ostrich</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>1.0</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>Zebra</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.0</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>Accuracy</p>
</blockquote></td>
<td></td>
<td></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>3.0</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>Macro Avg</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>3.0</p>
</blockquote></td>
</tr>
<tr>
<td><blockquote>
<p>Weighted Avg</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>0.00</p>
</blockquote></td>
<td><blockquote>
<p>3.0</p>
</blockquote></td>
</tr>
</tbody>
</table>

> Prediction for sample: Zebra

**RESULT :**

> Thus the above program is executed successfully.
