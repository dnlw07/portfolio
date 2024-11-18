

## Palantir Objects
- **Ontology** - a semantic layer that sits on top of multiple datasets (Data Mart?)
- **Object type** - dataset (Employee table)
- **Object set** - filtered sample of the data in a dataset (SQL query)
- **Object** - row in a dataset (single row with an employee name and ID)

## Notes
### Difference between `transfrom` vs `transform_df`
Difference between the `@transform` and `@transform_df` decorators:
- `@transform_df` operates exclusively on DataFrame objects. Sinlge output allowed.
- `@transform` operates on `transforms.api.TransformInput` and `transforms.api.TransformOutput` objects rather than DataFrames. **Multiple outputs are allowed.**

If your data transformation depends exclusively on DataFrame objects, you can use the `@transform_df()` decorator. This decorator injects DataFrame objects and expects the compute function to return a DataFrame.
Alternatively, you can use the more general `@transform()` decorator and explicitly call the `dataframe()` method to access a DataFrame containing your input dataset.

## Incremental computation
Incremental computation is an efficient method of generating output dataset. 
By leveraging the build history of a transform, incremental computation avoids the need to recompute the entire output dataset every time a transform runs.
When a transform is run incrementally, it means that records are appended into output dataset. Similarly, when a transform is run non-incrementally, it means that all existing records in the output are replaced by new set of records.

> [!WARNING]  Important!
> The compute function for your transform wrapped with the `incremental()` decorator must support being run both incrementally and non-incrementally.

There are two cases where the transform is allowed to run as a snapshot, even if `require_incremental=True`:
- One of the outputs has never been built before.
- The `semantic_version` has changed, meaning a snapshot was explicitly requested.

> [!WARNING]  Warning!
> Note that the Code Repositories preview feature will always run transforms in non-incremental mode. This is true even when `require_incremental=True` is passed into the `incremental()` decorator.

### Non-incremental example
In example below filter is performed over the entire input (rather than just the new data added). This is waste of compute resources and time.

```python
from transforms.api import transform, Input, Output
from pypsark.sql import functions as F, types as T, DataFrame

@transform(
    my_output=Output('/path/to/my/output'),
    students=Input('/path/to/my/input'),
)
def main(students, my_output):
    # Declare df
    students_df = students.dataframe() # default mode: current - (read all records from an input)

    # Logic
    output_df = students_df.filter(F.col('hair') != 'Brown')

    # Save data
    my_output.write_dataframe(output_df)

# Output:
# +---+-----+-----+------+                  +---+-----+-----+------+
# | id| hair|  eye|   sex|                  | id| hair|  eye|   sex|
# +---+-----+-----+------+     Build 1      +---+-----+-----+------+
# | 17|Black|Green|Female|    --------->    | 18|Brown|Green|Female|
# | 18|Brown|Green|Female|                  +---+-----+-----+------+
# | 19|  Red|Black|Female|
# +---+-----+-----+------+
# 
# ------------------------------------------------------------------
# 
# +---+-----+-----+------+                  +---+-----+-----+------+
# | id| hair|  eye|   sex|                  | id| hair|  eye|   sex|
# +---+-----+-----+------+     Build 2      +---+-----+-----+------+
# | 17|Black|Green|Female|    --------->    | 18|Brown|Green|Female|
# | 18|Brown|Green|Female|                  | 20|Brown|Amber|Female|
# | 19|  Red|Black|Female|                  +---+-----+-----+------+
# | 20|Brown|Amber|Female|
# | 21|Black|Blue |Male  |
# +---+-----+-----+------+
```

### Incremental example
In example below only **new records** from existing input dataset are taken (delta) and filtering is performed **only on the delta records**.
Afterwards the delta records are appended into existing output dataset.

```python
from transforms.api import transform, incremental, Input, Output
from pypsark.sql import functions as F, types as T, DataFrame

@incremental()
@transform(
    my_output=Output('/path/to/my/output'),
    students=Input('/path/to/my/input'),
)
def filter_eye_color(students, processed):
     # Declare df
    students_df = students.dataframe('added') # read delta records only (default behaviour for incremental read)

    # Logic
    students_df.filter(F.col('hair') == 'Brown')

   # Save data
    my_output.set_mode('modify') # append (default behaviour for incremental write)
    my_output.write_dataframe(output_df)

# Output:
# +---+-----+-----+------+                  +---+-----+-----+------+
# | id| hair|  eye|   sex|                  | id| hair|  eye|   sex|
# +---+-----+-----+------+     Build 1      +---+-----+-----+------+
# | 17|Black|Green|Female|    --------->    | 18|Brown|Green|Female|
# | 18|Brown|Green|Female|                  +---+-----+-----+------+
# | 19|  Red|Black|Female|
# +---+-----+-----+------+
# 
# ------------------------------------------------------------------
# 
# +---+-----+-----+------+     Build 2      +---+-----+-----+------+
# | 20|Brown|Amber|Female|    --------->    | 20|Brown|Amber|Female|
# | 21|Black|Blue |Male  |                  +---+-----+-----+------+
# +---+-----+-----+------+
```


### Incremental modes
> Source: https://www.palantir.com/docs/foundry/transforms-python/incremental-reference/

The `transforms.api.IncrementalTransformOutput` object provides access both to read and write modes for the output dataset.

#### Read modes
Read mode types for incremental Inputs and Outputs:

| Read Mode  | Incremental Behavior                                                                                                             | Non-incremental Behavior                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `added`    | Returns a DataFrame containing any new rows appended to the input since last time the transform ran. *(Default for incremental)* | Returns a DataFrame containing the entire dataset since all rows are considered unseen.                                                      |
| `previous` | Returns a DataFrame containing the entire input given to the transform the last time it ran.                                     | Returns an empty DataFrame.                                                                                                                  |
| `current`  | Returns a DataFrame containing the entire input dataset for the current run.                                                     | Returns a DataFrame containing the entire input dataset for the current run. This will be the same as added. *(Default for non-incremental)* |
##### Read mode for input
Default mode for incremental input is `added`. Whenever possible try to use `added` over `current` as it makes your intentions more clear.

##### Read mode for output
Although default read mode is `current`, in most cases you actually want to use `previous`. Other read modes should be used to read dataset after writing to it.


> [!WARNING]  Warning!
> The nature of incremental transforms means that we read all records form the input datasets since the last SNAPSHOT transaction. If you begin to see progressive slowness in your incremental transform, we recommend running a SNAPSHOT build on your incremental input datasets. Thanks that less records (smaller delta) will be loaded from input.

#### Write modes for output

| Write mode         | Behavior                                                                                                                                                                                                                               |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| modify<br>(append) | Modifies the existing output with data written during the build. Calling `write_dataframe()` method, when `my_output.set_mode('modify')` is set, will append the written DataFrame to the existing output. *(default for incremental)* |
| replace            | Fully replaces the output with data written during the build. *(default for non-incremental)*                                                                                                                                          |
#### IncrementalTransformContext
The `IncrementalTransformContext` object provides an additional property: `ctx.is_incremental`. This property is set to `True` if the transform is run incrementally, this means:
- the default inputs read mode is set to `added` and
- the default output write mode is set to `modify`.

##### Incremental only load
To read data from the previous output the transform must run in incremental mode (`ctx.is_incremental is True`), otherwise the dataframe will be empty.

Example:
```python
if ctx.is_incremental is True: 
    previous_output_df = my_output.dataframe('previous', output_schema)
```

> [!WARNING]  Warning!
> When using `set_mode()`, it is worth ensuring that this is valid behavior both scenarios when the transform is run incrementally or non-incrementally. If this is not the case, you should make use of the `ctx.is_incremental` property.
> 
> ```python
> # snippet to handle both incremental and non-incremental READ modes
> if ctx.is_incremental is True:
>     df = some_input.dataframe('added')
> else:
>     df = some_input.dataframe('current')
> 
> # snippet to handle both incremental and non-incremental WRITE modes
> if ctx.is_incremental is True:
>     df.set_mode('modify')
> else:
>     df.set_mode('replace')
> ```


#### Reading and writing data from the previous run - valid combinations

| Output Read Mode | Output Write Mode | Has new data been written yet? | Behavior |
|---|---|---|---|
| `current` |	`modify` | No | `dataframe()` will return the `previous` output of the transform.|
| `current` | `modify` | Yes | `dataframe()` will return the `previous` output of the transform + data written to output in currently running build.|
| ~~`current`~~ | ~~`replace`~~ | ~~No~~ | ~~There is no use case for these settings. Use `previous` mode instead.~~|
| `current` | `replace` | Yes | `dataframe()` will return data written to output in currently running build. |
| ~~`added`~~ | ~~`modify`/`replace`~~ | ~~No~~ | ~~There is no use case for these settings. Use `previous` mode instead.~~|
| `added` | `modify`/`replace` | Yes | `dataframe()` will return data written to output in currently running build.|
| `previous` | `modify` | Yes/No | `dataframe()` will return the `previous` output of the transform. |
| `previous` | `replace` | Yes/No | Returns an empty DataFrame. |

#### Snapshot inputs
There are scenarios in which it is allowed for inputs to be fully rewritten without invalidating the incrementality of the transform. For example, suppose you have a simple reference dataset that maps phone number country code to country and this is periodically rewritten. Changes to this dataset do not necessarily invalidate the results of any previous computation and therefore should not prevent the transform being run incrementally.
**By default, as described above, a transform cannot be run incrementally if any input has been fully rewritten since the transform was last run. Snapshot inputs are excluded from this check and their start transaction allowed to differ between runs.**
Snapshot inputs can be configured by using the `snapshot_inputs` argument on the `incremental()` decorator.

```python
@incremental(snapshot_inputs=['country_codes'])
@transform(
    phone_numbers=Input('/examples/phone_numbers'),
    country_codes=Input('/examples/country_codes'),
    output=Output('/examples/phone_numbers_to_country')
)
def map_phone_number_to_country(phone_numbers, country_codes, output):
    # type: (TransformInput, TransformInput, TransformOutput) -> None

    # this will be all unseen phone numbers since the previous run
    phone_numbers = phone_numbers.dataframe()
    # this will be all country codes, regardless of what has been seen previously
    country_codes = country_codes.dataframe()

    cond = [phone_numbers.country_code == country_codes.code]
    output.write_dataframe(phone_numbers.join(country_codes, on=cond, how='left_outer'))
```

The behavior of snapshot inputs are identical when a transform runs incrementally or non-incrementally. As such, `added` and `current` read modes will always return the entire dataset. All other read modes will return the empty dataset.

#### Changes to Inputs
The list of existing inputs can be modified. Incrementality will be preserved in the case where either:
- New inputs or new snapshot inputs are added, or
- Existing inputs or existing snapshot inputs are removed. Note that an incremental transform must have at least one input.

There is also a requirement that the start transactions for each of the non-snapshot input datasets are consistent with those used for the previous run.

#### Outputs last built by same transform
For multi-output incremental transforms, the last committed transaction on each of the outputs must have been generated from the same transform.

#### Summary of requirements for incremental computation
A transform can be run incrementally if and only if all its incremental inputs only had files appended to them, or where files were deleted, those files were deleted using Foundry Retention with allow_retention=True. Snapshot inputs are excluded from this check.

## Snippets
### Import libaries
```python
# palantir libaries
from transforms.api import transform_df, Input, Output
from transforms.api import transform, Input, Output
from transforms.api import transform, incremental, Input, Output
from transforms.api import configure

# pyspark libraries
from pypsark.sql import functions as F, types as T
from pyspark.sql import DataFrame, Row
from pyspark.sql.window import Window as W

# python
from re import replace, findall, match
from functools import reduce
```

### Single output
For signle output transformations use `@transform_df` decorator.

```python
from transforms.api import transform, Input, Output


@transform_df(
    Output("/output1"),
    df1=Input("/input1"),
    df2=Input("/input2")
)
def transformation(df1, df2):
    output_df = df1.join(df2, on='col1', how='left')
    return output_df
```



### Dynamic inputs for `@transform_df`
Generate dynamic inputs transformation for `@transform_df` decorator.

```python
from transforms.api import transform_df, Input, Output
from functools import reduce


datasets = ['dataset1', 'dataset2', 'dataset3']
inputs = {f'input_{i}': Input(f'input/folder/path/{dataset}') for dataset in datasets}
kwargs = {**{'output': Output('output/folder/path/unioned_dataset')}, **inputs}


@transform_df(**kwargs)
def compute_function(**inputs):
    unioned_df = reduce(Dataframe.unionByName, inputs)
    return unioned_df
```


### Multiple outputs
For transformations with multiple outputs use `@transform` decorator.

```python
from transforms.api import transform, Input, Output


@transform(
    output_1=Output("/output1"),
    output_2=Output("/output2"),
    input_1=Input("/input1"),
    input_2=Input("/input2")
)
def transformation(output1, output2, input1, input2):
    # convert into dataframe
    df1 = input_1.dataframe()
    df2 = input_2.dataframe()

    # wrtite output
    output_1.write_dataframe(df1)
    output_2.write_dataframe(df2)
```

### Dynamic inputs for `@transform`
Generate dynamic inputs and outputs for `@transform` decorator.

```python
from transforms.api import transform, Input, Output


datasets = {
    'output_1': Output("/output1"),
    'output_2': Output("/output2"),
    'input_1': Input("/input1"),
    'input_2': Input("/input2")
}


@transform(**datasets)
def transformation(ctx, **datasets):
    # convert into dataframe
    df1 = datasets['input_1'].dataframe()
    df2 = datasets['input_2'].dataframe()

    # wrtite output
    datasets['output_1'].write_dataframe(df1)
    datasets['output_2'].write_dataframe(df2)
```

### Incremental example
Example with use of incremental input and incremental output.
For incremental transformation use `@transform` decorator.

```python
from transforms.api import transform, Input, Output, incremental
from pypsark.sql import functions as F, types as T, DataFrame


# schema
OUTPUT_SCHEMA = T.StructType([
    T.StructField('date', T.DateType()),
    T.StructField('value', T.IntegerType())
])


# spark configuration
@configure(profile=[
    'EXECUTOR_MEMORY_MEDIUM',
    'NUM_EXECUTORS_32',
    'DRIVER_MEMORY_MEDIUM',
    'SHUFFLE_PARTITIONS_MEDIUM'
])
@incremental(semantic_version=1)
@transform(
    my_output=Output('/path/to/my/output'),
    my_input=Input('/path/to/my/input'),
)
def main(my_input, my_output):
    # Declare df
    previous_output_df = my_output.dataframe('previous', output_schema)
    current_input_df = my_input.dataframe('current')

    # Logic
    filtered_input = current_input_df.filter(F.col('date') != F.to_date(F.lit(None), 'yyyy-MM-dd'))
    output_df = filtered_input

    # Save data
    my_output.set_mode('modify') # append
    my_output.write_dataframe(output_df)
```


## Foundry issues

1. Data Lineage - build multiple datasets
> **Error:** Input dataset has no data. Please build it first.
> 
> **Solution:** 
> 1. set a proper fallback branch in Data Lineage and/or in Repository Settings
> 2. (workaround) Sometimes it is required to specify additional parameter `branch='branch_name'` for specified Inputs.


2. Inferring schema
> **Error:** You are referencing to the column `col_name` but it's missing form the schema.
> 
> **Reason:** Schema is inferred by spark, and column contains null values only. So its impossible for spark to infer the column type.
> 
> **Solution:** 
> 1. Replace `None` with empty string or `N/A`
> 2. Manually create schema


## Personal snippets

```python
## test_get_item.py
from transforms.api import transform_df, Output


@transform_df(
    Output("path/to/dataset"),
)
def main(ctx):
    '''
    description:
    getItem - methond is taking part of list or dict from column.
    the method is working directly on a COLUMN class, not dataframe class.
    '''

    # create test df
    input_df = ctx.spark_session.createDataFrame([([1, 2], {"key": "value"})], ["lst", "dct"])

    # incorrect syntax style
    # output_df = input_df.select(input_df.lst.getItem(0), input_df.dct.getItem("key"))

    # correct syntax style
    output_df = input_df.select(input_df['lst'].getItem(0), input_df['dct'].getItem("key"))

    return output_df


## Output example:
## input_df.show():
## +------+--------------+
## |   lst|           dct|
## +------+--------------+
## |[1, 2]|[key -> value]|
## +------+--------------+

## output_df.show():
## +------+--------+
## |lst[0]|dct[key]|
## +------+--------+
## |     1|   value|
## +------+--------+

#############################
## get column names by spcedific condition (data_type, or name)
#############################

from transforms.api import transform_df, Input, Output
from pyspark.sql import DataFrame
from functools import reduce
from ._const import MANUAL_INPUTS


## file_names = ['dataset_1', 'dataset_2']
file_names = [file_name for file_name, sub_dct in MANUAL_INPUTS.items()]  # ATTENTION! make sure that latest version of dictionary in 'filest_dict.py' is pasted from 'master' branch!

## dynamic input
inputs = {f'input_{num}': Input(f'/path/to/{file_name}', branch='master') for num, file_name in enumerate(file_names)}
output = Output("path/to/dataset")
kwargs = {**{'output': output}, **inputs}


@transform_df(**kwargs)
def compute_function(ctx, **inputs):
    '''
    function retruns list of columns for each dataset. column list we want to receive can be customized (example: return columns with type 'double').
    '''

    df_lst = []
    for input_num, input_df in inputs.items():
        num = int(input_num[6:])
        file_name = file_names[int(num)]
        column_names_lst = input_df.columns
        dtypes = input_df.dtypes

        filtered_column_names_lst = []
        if 'dataset_name' in file_name:
            # filter list of substring from list
            filtered_column_names_lst = [substr for substr in ['col_1', 'col_2', 'col_3'] if [col_name for col_name in column_names_lst if substr in col_name]]
        else:
            filtered_column_names_lst = [col_name for col_name, stype in dtypes if stype == 'double']

        # if list is empty infer schema error occurs - another solution is to provide schema for created df
        if len(filtered_column_names_lst) == 0:
            filtered_column_names_lst = ['empty']

        df = ctx.spark_session.createDataFrame([(file_name, filtered_column_names_lst)], ['df_name', 'col_lst'])
        df_lst.append(df)

    output_df = reduce(DataFrame.unionByName, df_lst)

    return output_df



#############################
## compare two datasets
#############################

from transforms.api import transform, Input, Output
from pyspark.sql import functions as F


'''
Compare two different branches of the same dataset.
1. Find columns causing discrepencies
2. Find PK for further join
3. Join two datasets to see diffrence between columns having discrepencies
'''

########################################
## CONTROL PANEL
order_of_precision = 4

branch_1 = 'branch_name_1'
dataset_1 = 'path/to/dataset'

branch_2 = 'branch_name_1'
dataset_2 = 'path/to/dataset'
########################################


exclude_col = [] # columns not included in minus analysis
key_cols = ['col_1']
exception_list = [] # columns filled with nulls etc.
value_cols = [] # if empty then takes columns based on df schema

global datasets
datasets = {
    'out_minus_all_cols': Output('path/to/dataset/1_minus_all_cols'),
    'out_excluded_minus': Output('path/to/dataset/2_excluded_minus'),
    'out_filtered_joined': Output('path/to/dataset/3_filtered_joined'),
    'out_number_mismatch': Output('path/to/dataset/4_number_mismatch'),
    'out_summary': Output('path/to/dataset/5_summary'),
    'input_1': Input(dataset_1, branch=branch_1),
    'input_2': Input(dataset_2, branch=branch_2),
}


@transform(**datasets)
def compute_function(ctx, **datasets):

    # dynamically declare inputs into dataframes (compiler will show errors for dynamically declared variables)
    # [globals().update({f'df{num+1}': val.dataframe()}) for num, (key, val) in enumerate(datasets.items()) if 'input_' in key]

    df1 = datasets['input_1'].dataframe()
    df2 = datasets['input_2'].dataframe()

    value_cols = [c for c, t in df1.dtypes if t in ['integer', 'double', 'float', 'decimal']]

    ########################################
    # 1. check if branches are the same - if yes remove columns causing discrepancies
    ########################################

    # minus all columnus
    df1_subtract = df1.subtract(df2)
    df2_subtract = df2.subtract(df1)

    df1_subtract = df1_subtract.select(F.col('*'), F.lit('df1 - df2').alias('df'))
    df2_subtract = df2_subtract.select(F.col('*'), F.lit('df2 - df1').alias('df'))

    minus_all_cols_df = df1_subtract.unionByName(df2_subtract)

    # minus selected columnus only
    exclude_col = ['target_file_id', 'df']
    df1_excluded = df1.drop(*exclude_col)
    df2_excluded = df2.drop(*exclude_col)

    df1_subtract = df1_excluded.subtract(df2_excluded)
    df2_subtract = df2_excluded.subtract(df1_excluded)

    df1_subtract = df1_subtract.select(F.col('*'), F.lit('df1 - df2').alias('df'))
    df2_subtract = df2_subtract.select(F.col('*'), F.lit('df2 - df1').alias('df'))

    excluded_minus_df = df1_subtract.unionByName(df2_subtract)

    # ########################################
    # # 2. find PK fields
    # ########################################

    df1 = df1.createOrReplaceTempView('df1')

    cols = ['col_1', 'col_2']

    # check PK uniqueness
    # output_df = ctx.spark_session.sql(f'select col_1, col_2, count(*) as duplicates from df1 group bycol_1, col_2 having count(*)>1')

    # ########################################
    # # 3. compare columns having discrepancies
    # ########################################
    join_condition = [df1[c].eqNullSafe(df2[c]) for c in key_cols]
    joined_df = df1.alias('left').join(df2.alias('right'), join_condition, 'inner').select(
        *[F.col(f'left.{c}') for c in key_cols if c not in value_cols],
        *[F.col(f'left.{c}').alias(f'left_{c}') for c in df1.columns if c in value_cols],
        *[F.col(f'right.{c}').alias(f'right_{c}') for c in df2.columns if c in value_cols],
        *[F.col(f'left.{c}').alias(f'left_{c}') for c in df1.columns if c not in key_cols and c not in exception_list],
        *[F.col(f'right.{c}').alias(f'right_{c}') for c in df2.columns if c not in key_cols and c not in exception_list]
    )

    # filter columns with discrepencies
    filtered_joined_df = joined_df.select('left_col_1', 'right_col_1').filter(F.col('left_col_1') != F.col('right_col_1'))

    # aggregate data by count
    agg_filtered_joined_df = filtered_joined_df.groupby('left_col_1', 'right_col_1').count()

    # ########################################
    # # 4. compare number columns
    # ########################################
    # list of number columns in the dataframe

    # create columns storing subtraction of each number column pair
    number_check_df = joined_df.select([(F.abs(F.col(f'left_{c}') - F.col(f'right_{c}'))).alias(f'diff_{c}') for c in value_cols])

    # filter values above the treshold
    number_mismatch_df = number_check_df.filter((sum([F.col(f'diff_{c}') for c in value_cols]) > (10**(-order_of_precision))))

    ########################################
    # 5. summary
    ########################################
    summary_data = [
        (1, 'df1 count:', str(df1.count())),
        (2, 'df2 count:', str(df2.count())),
        (3, 'df1_excluded count:', str(df1_excluded.count())),
        (4, 'df2_excluded count:', str(df2_excluded.count())),
        (5, 'joined rows:', str(joined_df.count())),
        (6, 'rows with discrepencies:', str(filtered_joined_df.count())),
        (7, 'ratio of incorrect rows in joned dataset:', f'{str(filtered_joined_df.count()/joined_df.count()*100)}%'),
        (8, 'is join correct?', 'YES' if (joined_df.count() == min(df1.count(), df2.count())) else 'NO')
    ]
    summary_df = ctx.spark_session.createDataFrame(summary_data, ['Id', 'Key', 'Value'])
    summary_df.orderBy('Id')



    # write dataframes
    locals_dct = locals() # pass funcion locals() into comprehension list
    [val.write_dataframe(locals_dct[f'{key[4:]}_df']) for key, val in datasets.items() if 'out_' in key]

```