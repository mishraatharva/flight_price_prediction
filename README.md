FLIGHT PRICE PREDICTION USING AWS-SAGEMAKER

rename()
assign() -> to create new columns and update existing columns
str.contains() # parameter regular expression


 if: tail is high and pick is high -> excess kurtosis
 if: small tail and hight is flat as well -> -ve kurtosis
 if: normaal distribution -> kurtosis close to zero


-----------------------------------------------------------------------
NOTE:
-----------------------------------------------------------------------
np.select() expect condition to be boolean array not integer.
so we pass train["columns__name"].ge(150) as an input,
not train["duration"].ge(150).astype("int").

becoz:
- train["columns_name"].ge(150).astype("int") : returns integer array
- train["columns_name"].ge(150) : returns boolean array