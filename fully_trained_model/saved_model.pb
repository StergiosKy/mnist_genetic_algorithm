·È
é¾
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ª
|
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_15/kernel
u
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel* 
_output_shapes
:
*
dtype0
s
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_15/bias
l
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
_output_shapes	
:*
dtype0
|
dense_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_16/kernel
u
#dense_16/kernel/Read/ReadVariableOpReadVariableOpdense_16/kernel* 
_output_shapes
:
*
dtype0
s
dense_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_16/bias
l
!dense_16/bias/Read/ReadVariableOpReadVariableOpdense_16/bias*
_output_shapes	
:*
dtype0
{
dense_17/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
* 
shared_namedense_17/kernel
t
#dense_17/kernel/Read/ReadVariableOpReadVariableOpdense_17/kernel*
_output_shapes
:	
*
dtype0
r
dense_17/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_17/bias
k
!dense_17/bias/Read/ReadVariableOpReadVariableOpdense_17/bias*
_output_shapes
:
*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
ï
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ª
value B B
æ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
h


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
6
iter
	decay
learning_rate
momentum
*

0
1
2
3
4
5
*

0
1
2
3
4
5
 
­
trainable_variables
 metrics
!layer_metrics
"non_trainable_variables
	variables
#layer_regularization_losses

$layers
regularization_losses
 
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1


0
1
 
­
trainable_variables
%metrics
&layer_metrics
'non_trainable_variables
	variables
(layer_regularization_losses

)layers
regularization_losses
[Y
VARIABLE_VALUEdense_16/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_16/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
*metrics
+layer_metrics
,non_trainable_variables
	variables
-layer_regularization_losses

.layers
regularization_losses
[Y
VARIABLE_VALUEdense_17/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_17/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
/metrics
0layer_metrics
1non_trainable_variables
	variables
2layer_regularization_losses

3layers
regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE

40
51
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	6total
	7count
8	variables
9	keras_api
D
	:total
	;count
<
_fn_kwargs
=	variables
>	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

60
71

8	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

=	variables

serving_default_dense_15_inputPlaceholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
¨
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_15_inputdense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_133331
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ú
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOp#dense_16/kernel/Read/ReadVariableOp!dense_16/bias/Read/ReadVariableOp#dense_17/kernel/Read/ReadVariableOp!dense_17/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_133680
Ý
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_15/kerneldense_15/biasdense_16/kerneldense_16/biasdense_17/kerneldense_17/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_133732Ôë
¿
Ã
D__inference_dense_16_layer_call_and_return_conditional_losses_133050

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÇ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul¿
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mulþ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
~
)__inference_dense_15_layer_call_fn_133507

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1330112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

__inference_loss_fn_1_133593<
8dense_15_bias_regularizer_square_readvariableop_resource
identity¢/dense_15/bias/Regularizer/Square/ReadVariableOpØ
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_15_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mul
IdentityIdentity!dense_15/bias/Regularizer/mul:z:00^dense_15/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp
ï

__inference_loss_fn_3_133615<
8dense_16_bias_regularizer_square_readvariableop_resource
identity¢/dense_16/bias/Regularizer/Square/ReadVariableOpØ
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp8dense_16_bias_regularizer_square_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul
IdentityIdentity!dense_16/bias/Regularizer/mul:z:00^dense_16/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp
&
¾
!__inference__wrapped_model_132984
dense_15_input8
4sequential_5_dense_15_matmul_readvariableop_resource9
5sequential_5_dense_15_biasadd_readvariableop_resource8
4sequential_5_dense_16_matmul_readvariableop_resource9
5sequential_5_dense_16_biasadd_readvariableop_resource8
4sequential_5_dense_17_matmul_readvariableop_resource9
5sequential_5_dense_17_biasadd_readvariableop_resource
identity¢,sequential_5/dense_15/BiasAdd/ReadVariableOp¢+sequential_5/dense_15/MatMul/ReadVariableOp¢,sequential_5/dense_16/BiasAdd/ReadVariableOp¢+sequential_5/dense_16/MatMul/ReadVariableOp¢,sequential_5/dense_17/BiasAdd/ReadVariableOp¢+sequential_5/dense_17/MatMul/ReadVariableOpÑ
+sequential_5/dense_15/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+sequential_5/dense_15/MatMul/ReadVariableOp¾
sequential_5/dense_15/MatMulMatMuldense_15_input3sequential_5/dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_15/MatMulÏ
,sequential_5/dense_15/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,sequential_5/dense_15/BiasAdd/ReadVariableOpÚ
sequential_5/dense_15/BiasAddBiasAdd&sequential_5/dense_15/MatMul:product:04sequential_5/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_15/BiasAdd
sequential_5/dense_15/ReluRelu&sequential_5/dense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_15/ReluÑ
+sequential_5/dense_16/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02-
+sequential_5/dense_16/MatMul/ReadVariableOpØ
sequential_5/dense_16/MatMulMatMul(sequential_5/dense_15/Relu:activations:03sequential_5/dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_16/MatMulÏ
,sequential_5/dense_16/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,sequential_5/dense_16/BiasAdd/ReadVariableOpÚ
sequential_5/dense_16/BiasAddBiasAdd&sequential_5/dense_16/MatMul:product:04sequential_5/dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_16/BiasAdd
sequential_5/dense_16/ReluRelu&sequential_5/dense_16/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_5/dense_16/ReluÐ
+sequential_5/dense_17/MatMul/ReadVariableOpReadVariableOp4sequential_5_dense_17_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02-
+sequential_5/dense_17/MatMul/ReadVariableOp×
sequential_5/dense_17/MatMulMatMul(sequential_5/dense_16/Relu:activations:03sequential_5/dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_5/dense_17/MatMulÎ
,sequential_5/dense_17/BiasAdd/ReadVariableOpReadVariableOp5sequential_5_dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02.
,sequential_5/dense_17/BiasAdd/ReadVariableOpÙ
sequential_5/dense_17/BiasAddBiasAdd&sequential_5/dense_17/MatMul:product:04sequential_5/dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_5/dense_17/BiasAdd£
sequential_5/dense_17/SoftmaxSoftmax&sequential_5/dense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
sequential_5/dense_17/Softmax
IdentityIdentity'sequential_5/dense_17/Softmax:softmax:0-^sequential_5/dense_15/BiasAdd/ReadVariableOp,^sequential_5/dense_15/MatMul/ReadVariableOp-^sequential_5/dense_16/BiasAdd/ReadVariableOp,^sequential_5/dense_16/MatMul/ReadVariableOp-^sequential_5/dense_17/BiasAdd/ReadVariableOp,^sequential_5/dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2\
,sequential_5/dense_15/BiasAdd/ReadVariableOp,sequential_5/dense_15/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_15/MatMul/ReadVariableOp+sequential_5/dense_15/MatMul/ReadVariableOp2\
,sequential_5/dense_16/BiasAdd/ReadVariableOp,sequential_5/dense_16/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_16/MatMul/ReadVariableOp+sequential_5/dense_16/MatMul/ReadVariableOp2\
,sequential_5/dense_17/BiasAdd/ReadVariableOp,sequential_5/dense_17/BiasAdd/ReadVariableOp2Z
+sequential_5/dense_17/MatMul/ReadVariableOp+sequential_5/dense_17/MatMul/ReadVariableOp:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_15_input

Æ
-__inference_sequential_5_layer_call_fn_133222
dense_15_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCalldense_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1332072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_15_input
°
£
__inference_loss_fn_0_133582>
:dense_15_kernel_regularizer_square_readvariableop_resource
identity¢1dense_15/kernel/Regularizer/Square/ReadVariableOpã
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_15_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul
IdentityIdentity#dense_15/kernel/Regularizer/mul:z:02^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp
ê
¾
-__inference_sequential_5_layer_call_fn_133446

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1332072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
Ã
D__inference_dense_15_layer_call_and_return_conditional_losses_133498

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÇ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul¿
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mulþ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
&
Ô
__inference__traced_save_133680
file_prefix.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop.
*savev2_dense_16_kernel_read_readvariableop,
(savev2_dense_16_bias_read_readvariableop.
*savev2_dense_17_kernel_read_readvariableop,
(savev2_dense_17_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameý
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¦
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesø
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop*savev2_dense_16_kernel_read_readvariableop(savev2_dense_16_bias_read_readvariableop*savev2_dense_17_kernel_read_readvariableop(savev2_dense_17_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*^
_input_shapesM
K: :
::
::	
:
: : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¾8
¢
H__inference_sequential_5_layer_call_and_return_conditional_losses_133118
dense_15_input
dense_15_133022
dense_15_133024
dense_16_133061
dense_16_133063
dense_17_133088
dense_17_133090
identity¢ dense_15/StatefulPartitionedCall¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall 
 dense_15/StatefulPartitionedCallStatefulPartitionedCalldense_15_inputdense_15_133022dense_15_133024*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1330112"
 dense_15/StatefulPartitionedCall»
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_133061dense_16_133063*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1330502"
 dense_16/StatefulPartitionedCallº
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_133088dense_17_133090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1330772"
 dense_17/StatefulPartitionedCall¸
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133022* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul¯
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133024*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mul¸
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133061* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul¯
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133063*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul²
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_15_input
¿
Ã
D__inference_dense_16_layer_call_and_return_conditional_losses_133542

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÇ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul¿
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mulþ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
~
)__inference_dense_17_layer_call_fn_133571

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1330772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦8

H__inference_sequential_5_layer_call_and_return_conditional_losses_133207

inputs
dense_15_133167
dense_15_133169
dense_16_133172
dense_16_133174
dense_17_133177
dense_17_133179
identity¢ dense_15/StatefulPartitionedCall¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_133167dense_15_133169*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1330112"
 dense_15/StatefulPartitionedCall»
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_133172dense_16_133174*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1330502"
 dense_16/StatefulPartitionedCallº
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_133177dense_17_133179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1330772"
 dense_17/StatefulPartitionedCall¸
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133167* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul¯
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133169*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mul¸
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133172* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul¯
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133174*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul²
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦8

H__inference_sequential_5_layer_call_and_return_conditional_losses_133267

inputs
dense_15_133227
dense_15_133229
dense_16_133232
dense_16_133234
dense_17_133237
dense_17_133239
identity¢ dense_15/StatefulPartitionedCall¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall
 dense_15/StatefulPartitionedCallStatefulPartitionedCallinputsdense_15_133227dense_15_133229*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1330112"
 dense_15/StatefulPartitionedCall»
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_133232dense_16_133234*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1330502"
 dense_16/StatefulPartitionedCallº
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_133237dense_17_133239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1330772"
 dense_17/StatefulPartitionedCall¸
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133227* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul¯
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133229*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mul¸
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133232* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul¯
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133234*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul²
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾8
¢
H__inference_sequential_5_layer_call_and_return_conditional_losses_133161
dense_15_input
dense_15_133121
dense_15_133123
dense_16_133126
dense_16_133128
dense_17_133131
dense_17_133133
identity¢ dense_15/StatefulPartitionedCall¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢ dense_16/StatefulPartitionedCall¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢ dense_17/StatefulPartitionedCall 
 dense_15/StatefulPartitionedCallStatefulPartitionedCalldense_15_inputdense_15_133121dense_15_133123*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_15_layer_call_and_return_conditional_losses_1330112"
 dense_15/StatefulPartitionedCall»
 dense_16/StatefulPartitionedCallStatefulPartitionedCall)dense_15/StatefulPartitionedCall:output:0dense_16_133126dense_16_133128*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1330502"
 dense_16/StatefulPartitionedCallº
 dense_17/StatefulPartitionedCallStatefulPartitionedCall)dense_16/StatefulPartitionedCall:output:0dense_17_133131dense_17_133133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_17_layer_call_and_return_conditional_losses_1330772"
 dense_17/StatefulPartitionedCall¸
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133121* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul¯
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_15_133123*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mul¸
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133126* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul¯
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOpdense_16_133128*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul²
IdentityIdentity)dense_17/StatefulPartitionedCall:output:0!^dense_15/StatefulPartitionedCall0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp!^dense_16/StatefulPartitionedCall0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp!^dense_17/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2D
 dense_16/StatefulPartitionedCall dense_16/StatefulPartitionedCall2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2D
 dense_17/StatefulPartitionedCall dense_17/StatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_15_input
ù	
Ý
D__inference_dense_17_layer_call_and_return_conditional_losses_133077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿E

H__inference_sequential_5_layer_call_and_return_conditional_losses_133380

inputs+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOpª
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¨
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¦
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/Reluª
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_16/MatMul/ReadVariableOp¤
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul¨
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp¦
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/Relu©
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02 
dense_17/MatMul/ReadVariableOp£
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/SoftmaxÐ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulÈ
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mulÐ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulÈ
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
-__inference_sequential_5_layer_call_fn_133282
dense_15_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCalldense_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1332672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_15_input
<
ë
"__inference__traced_restore_133732
file_prefix$
 assignvariableop_dense_15_kernel$
 assignvariableop_1_dense_15_bias&
"assignvariableop_2_dense_16_kernel$
 assignvariableop_3_dense_16_bias&
"assignvariableop_4_dense_17_kernel$
 assignvariableop_5_dense_17_bias
assignvariableop_6_sgd_iter 
assignvariableop_7_sgd_decay(
$assignvariableop_8_sgd_learning_rate#
assignvariableop_9_sgd_momentum
assignvariableop_10_total
assignvariableop_11_count
assignvariableop_12_total_1
assignvariableop_13_count_1
identity_15¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¬
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesö
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_dense_15_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_15_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_16_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¥
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_16_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4§
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_17_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¥
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_17_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6 
AssignVariableOp_6AssignVariableOpassignvariableop_6_sgd_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¡
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8©
AssignVariableOp_8AssignVariableOp$assignvariableop_8_sgd_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_sgd_momentumIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¡
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¡
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12£
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13£
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ã
~
)__inference_dense_16_layer_call_fn_133551

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_16_layer_call_and_return_conditional_losses_1330502
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
£
__inference_loss_fn_2_133604>
:dense_16_kernel_regularizer_square_readvariableop_resource
identity¢1dense_16/kernel/Regularizer/Square/ReadVariableOpã
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp:dense_16_kernel_regularizer_square_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mul
IdentityIdentity#dense_16/kernel/Regularizer/mul:z:02^dense_16/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp
Ò
½
$__inference_signature_wrapper_133331
dense_15_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_15_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_1329842
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_15_input
ù	
Ý
D__inference_dense_17_layer_call_and_return_conditional_losses_133562

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
Softmax
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿E

H__inference_sequential_5_layer_call_and_return_conditional_losses_133429

inputs+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource+
'dense_16_matmul_readvariableop_resource,
(dense_16_biasadd_readvariableop_resource+
'dense_17_matmul_readvariableop_resource,
(dense_17_biasadd_readvariableop_resource
identity¢dense_15/BiasAdd/ReadVariableOp¢dense_15/MatMul/ReadVariableOp¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp¢dense_16/BiasAdd/ReadVariableOp¢dense_16/MatMul/ReadVariableOp¢/dense_16/bias/Regularizer/Square/ReadVariableOp¢1dense_16/kernel/Regularizer/Square/ReadVariableOp¢dense_17/BiasAdd/ReadVariableOp¢dense_17/MatMul/ReadVariableOpª
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_15/MatMul/ReadVariableOp
dense_15/MatMulMatMulinputs&dense_15/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/MatMul¨
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_15/BiasAdd/ReadVariableOp¦
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/BiasAddt
dense_15/ReluReludense_15/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_15/Reluª
dense_16/MatMul/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
dense_16/MatMul/ReadVariableOp¤
dense_16/MatMulMatMuldense_15/Relu:activations:0&dense_16/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/MatMul¨
dense_16/BiasAdd/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_16/BiasAdd/ReadVariableOp¦
dense_16/BiasAddBiasAdddense_16/MatMul:product:0'dense_16/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/BiasAddt
dense_16/ReluReludense_16/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_16/Relu©
dense_17/MatMul/ReadVariableOpReadVariableOp'dense_17_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02 
dense_17/MatMul/ReadVariableOp£
dense_17/MatMulMatMuldense_16/Relu:activations:0&dense_17/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/MatMul§
dense_17/BiasAdd/ReadVariableOpReadVariableOp(dense_17_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_17/BiasAdd/ReadVariableOp¥
dense_17/BiasAddBiasAdddense_17/MatMul:product:0'dense_17/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/BiasAdd|
dense_17/SoftmaxSoftmaxdense_17/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_17/SoftmaxÐ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mulÈ
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mulÐ
1dense_16/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'dense_16_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_16/kernel/Regularizer/Square/ReadVariableOp¸
"dense_16/kernel/Regularizer/SquareSquare9dense_16/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_16/kernel/Regularizer/Square
!dense_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_16/kernel/Regularizer/Const¾
dense_16/kernel/Regularizer/SumSum&dense_16/kernel/Regularizer/Square:y:0*dense_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/Sum
!dense_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_16/kernel/Regularizer/mul/xÀ
dense_16/kernel/Regularizer/mulMul*dense_16/kernel/Regularizer/mul/x:output:0(dense_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_16/kernel/Regularizer/mulÈ
/dense_16/bias/Regularizer/Square/ReadVariableOpReadVariableOp(dense_16_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_16/bias/Regularizer/Square/ReadVariableOp­
 dense_16/bias/Regularizer/SquareSquare7dense_16/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_16/bias/Regularizer/Square
dense_16/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_16/bias/Regularizer/Const¶
dense_16/bias/Regularizer/SumSum$dense_16/bias/Regularizer/Square:y:0(dense_16/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/Sum
dense_16/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_16/bias/Regularizer/mul/x¸
dense_16/bias/Regularizer/mulMul(dense_16/bias/Regularizer/mul/x:output:0&dense_16/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_16/bias/Regularizer/mul
IdentityIdentitydense_17/Softmax:softmax:0 ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp ^dense_16/BiasAdd/ReadVariableOp^dense_16/MatMul/ReadVariableOp0^dense_16/bias/Regularizer/Square/ReadVariableOp2^dense_16/kernel/Regularizer/Square/ReadVariableOp ^dense_17/BiasAdd/ReadVariableOp^dense_17/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp2B
dense_16/BiasAdd/ReadVariableOpdense_16/BiasAdd/ReadVariableOp2@
dense_16/MatMul/ReadVariableOpdense_16/MatMul/ReadVariableOp2b
/dense_16/bias/Regularizer/Square/ReadVariableOp/dense_16/bias/Regularizer/Square/ReadVariableOp2f
1dense_16/kernel/Regularizer/Square/ReadVariableOp1dense_16/kernel/Regularizer/Square/ReadVariableOp2B
dense_17/BiasAdd/ReadVariableOpdense_17/BiasAdd/ReadVariableOp2@
dense_17/MatMul/ReadVariableOpdense_17/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
¾
-__inference_sequential_5_layer_call_fn_133463

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_5_layer_call_and_return_conditional_losses_1332672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
Ã
D__inference_dense_15_layer_call_and_return_conditional_losses_133011

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢/dense_15/bias/Regularizer/Square/ReadVariableOp¢1dense_15/kernel/Regularizer/Square/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ReluÇ
1dense_15/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype023
1dense_15/kernel/Regularizer/Square/ReadVariableOp¸
"dense_15/kernel/Regularizer/SquareSquare9dense_15/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
2$
"dense_15/kernel/Regularizer/Square
!dense_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2#
!dense_15/kernel/Regularizer/Const¾
dense_15/kernel/Regularizer/SumSum&dense_15/kernel/Regularizer/Square:y:0*dense_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/Sum
!dense_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_15/kernel/Regularizer/mul/xÀ
dense_15/kernel/Regularizer/mulMul*dense_15/kernel/Regularizer/mul/x:output:0(dense_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_15/kernel/Regularizer/mul¿
/dense_15/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype021
/dense_15/bias/Regularizer/Square/ReadVariableOp­
 dense_15/bias/Regularizer/SquareSquare7dense_15/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes	
:2"
 dense_15/bias/Regularizer/Square
dense_15/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
dense_15/bias/Regularizer/Const¶
dense_15/bias/Regularizer/SumSum$dense_15/bias/Regularizer/Square:y:0(dense_15/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/Sum
dense_15/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense_15/bias/Regularizer/mul/x¸
dense_15/bias/Regularizer/mulMul(dense_15/bias/Regularizer/mul/x:output:0&dense_15/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense_15/bias/Regularizer/mulþ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp0^dense_15/bias/Regularizer/Square/ReadVariableOp2^dense_15/kernel/Regularizer/Square/ReadVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2b
/dense_15/bias/Regularizer/Square/ReadVariableOp/dense_15/bias/Regularizer/Square/ReadVariableOp2f
1dense_15/kernel/Regularizer/Square/ReadVariableOp1dense_15/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*º
serving_default¦
J
dense_15_input8
 serving_default_dense_15_input:0ÿÿÿÿÿÿÿÿÿ<
dense_170
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:Õ
®%
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api
	
signatures
?__call__
*@&call_and_return_all_conditional_losses
A_default_save_signature"î"
_tf_keras_sequentialÏ"{"class_name": "Sequential", "name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_15_input"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 397, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 397, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_15_input"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 397, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 397, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
¸	


kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layerù{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 784]}, "dtype": "float32", "units": 397, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 784]}}
Ã

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Dense", "name": "dense_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_16", "trainable": true, "dtype": "float32", "units": 397, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "bias_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 397}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 397]}}
÷

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
F__call__
*G&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_17", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 397}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 397]}}
I
iter
	decay
learning_rate
momentum"
	optimizer
J

0
1
2
3
4
5"
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
<
H0
I1
J2
K3"
trackable_list_wrapper
Ê
trainable_variables
 metrics
!layer_metrics
"non_trainable_variables
	variables
#layer_regularization_losses

$layers
regularization_losses
?__call__
A_default_save_signature
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
,
Lserving_default"
signature_map
#:!
2dense_15/kernel
:2dense_15/bias
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
­
trainable_variables
%metrics
&layer_metrics
'non_trainable_variables
	variables
(layer_regularization_losses

)layers
regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
#:!
2dense_16/kernel
:2dense_16/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
­
trainable_variables
*metrics
+layer_metrics
,non_trainable_variables
	variables
-layer_regularization_losses

.layers
regularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
": 	
2dense_17/kernel
:
2dense_17/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
/metrics
0layer_metrics
1non_trainable_variables
	variables
2layer_regularization_losses

3layers
regularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
.
40
51"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
»
	6total
	7count
8	variables
9	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
ÿ
	:total
	;count
<
_fn_kwargs
=	variables
>	keras_api"¸
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
60
71"
trackable_list_wrapper
-
8	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
-
=	variables"
_generic_user_object
2ÿ
-__inference_sequential_5_layer_call_fn_133222
-__inference_sequential_5_layer_call_fn_133446
-__inference_sequential_5_layer_call_fn_133282
-__inference_sequential_5_layer_call_fn_133463À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_sequential_5_layer_call_and_return_conditional_losses_133118
H__inference_sequential_5_layer_call_and_return_conditional_losses_133429
H__inference_sequential_5_layer_call_and_return_conditional_losses_133380
H__inference_sequential_5_layer_call_and_return_conditional_losses_133161À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
!__inference__wrapped_model_132984¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
dense_15_inputÿÿÿÿÿÿÿÿÿ
Ó2Ð
)__inference_dense_15_layer_call_fn_133507¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_15_layer_call_and_return_conditional_losses_133498¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_16_layer_call_fn_133551¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_16_layer_call_and_return_conditional_losses_133542¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_dense_17_layer_call_fn_133571¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_17_layer_call_and_return_conditional_losses_133562¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
³2°
__inference_loss_fn_0_133582
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_1_133593
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_2_133604
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
³2°
__inference_loss_fn_3_133615
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÒBÏ
$__inference_signature_wrapper_133331dense_15_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!__inference__wrapped_model_132984w
8¢5
.¢+
)&
dense_15_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ
¦
D__inference_dense_15_layer_call_and_return_conditional_losses_133498^
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_15_layer_call_fn_133507Q
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dense_16_layer_call_and_return_conditional_losses_133542^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dense_16_layer_call_fn_133551Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¥
D__inference_dense_17_layer_call_and_return_conditional_losses_133562]0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 }
)__inference_dense_17_layer_call_fn_133571P0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
;
__inference_loss_fn_0_133582
¢

¢ 
ª " ;
__inference_loss_fn_1_133593¢

¢ 
ª " ;
__inference_loss_fn_2_133604¢

¢ 
ª " ;
__inference_loss_fn_3_133615¢

¢ 
ª " ½
H__inference_sequential_5_layer_call_and_return_conditional_losses_133118q
@¢=
6¢3
)&
dense_15_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ½
H__inference_sequential_5_layer_call_and_return_conditional_losses_133161q
@¢=
6¢3
)&
dense_15_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
H__inference_sequential_5_layer_call_and_return_conditional_losses_133380i
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 µ
H__inference_sequential_5_layer_call_and_return_conditional_losses_133429i
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
-__inference_sequential_5_layer_call_fn_133222d
@¢=
6¢3
)&
dense_15_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

-__inference_sequential_5_layer_call_fn_133282d
@¢=
6¢3
)&
dense_15_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ

-__inference_sequential_5_layer_call_fn_133446\
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ

-__inference_sequential_5_layer_call_fn_133463\
8¢5
.¢+
!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
²
$__inference_signature_wrapper_133331
J¢G
¢ 
@ª=
;
dense_15_input)&
dense_15_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_17"
dense_17ÿÿÿÿÿÿÿÿÿ
