��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
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
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28��
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
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
�
mean_aggregation_layer_1_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *0
shared_name!mean_aggregation_layer_1_weight
�
3mean_aggregation_layer_1_weight/Read/ReadVariableOpReadVariableOpmean_aggregation_layer_1_weight*
_output_shapes

:@ *
dtype0
�
mean_aggregation_layer_2_weightVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@ *0
shared_name!mean_aggregation_layer_2_weight
�
3mean_aggregation_layer_2_weight/Read/ReadVariableOpReadVariableOpmean_aggregation_layer_2_weight*
_output_shapes

:@ *
dtype0

NoOpNoOp
�

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�	
value�	B�	 B�	
V
samples

seq_layers
	dense
	optimizer
	keras_api

signatures
 

0
1
^

	kernel

	variables
trainable_variables
regularization_losses
	keras_api
6
iter
	decay
learning_rate
momentum
 
 
;
#mean_aggregation_layer_1_weight
w
	keras_api
;
#mean_aggregation_layer_2_weight
w
	keras_api
IG
VARIABLE_VALUEdense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE

	0

	0
 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics

	variables
trainable_variables
regularization_losses
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEmean_aggregation_layer_1_weightGseq_layers/0/mean_aggregation_layer_1_weight/.ATTRIBUTES/VARIABLE_VALUE
 
|z
VARIABLE_VALUEmean_aggregation_layer_2_weightGseq_layers/1/mean_aggregation_layer_2_weight/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
 
k
call_dstPlaceholder*'
_output_shapes
:��������� *
dtype0*
shape:��������� 
�
call_dst_negPlaceholder*4
_output_shapes"
 :������������������ *
dtype0*)
shape :������������������ 
�
call_dst_neg_negPlaceholder*A
_output_shapes/
-:+��������������������������� *
dtype0*6
shape-:+��������������������������� 
k
call_srcPlaceholder*'
_output_shapes
:��������� *
dtype0*
shape:��������� 
�
call_src_negPlaceholder*4
_output_shapes"
 :������������������ *
dtype0*)
shape :������������������ 
�
call_src_neg_negPlaceholder*A
_output_shapes/
-:+��������������������������� *
dtype0*6
shape-:+��������������������������� 
�
StatefulPartitionedCallStatefulPartitionedCallcall_dstcall_dst_negcall_dst_neg_negcall_srccall_src_negcall_src_neg_negmean_aggregation_layer_2_weightmean_aggregation_layer_1_weightdense/kernel*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference_signature_wrapper_15736
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp3mean_aggregation_layer_1_weight/Read/ReadVariableOp3mean_aggregation_layer_2_weight/Read/ReadVariableOpConst*
Tin
2		*
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
GPU2*0J 8� *'
f"R 
__inference__traced_save_16197
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernelSGD/iter	SGD/decaySGD/learning_rateSGD/momentummean_aggregation_layer_1_weightmean_aggregation_layer_2_weight*
Tin

2*
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
GPU2*0J 8� **
f%R#
!__inference__traced_restore_16228��
؟
�
__inference_train_16126
src
src_neg
src_neg_neg
dst
dst_neg
dst_neg_neg

labels
unknown:@ 
	unknown_0:@ 
	unknown_1:@*
 sgd_cast_readvariableop_resource: ,
"sgd_cast_1_readvariableop_resource: .
$sgd_sgd_assignaddvariableop_resource:	 
identity��SGD/Cast/ReadVariableOp�SGD/Cast_1/ReadVariableOp�SGD/SGD/AssignAddVariableOp�+SGD/SGD/update/ResourceApplyGradientDescent�-SGD/SGD/update_1/ResourceApplyGradientDescent�-SGD/SGD/update_2/ResourceApplyGradientDescent�StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsrcsrc_negsrc_neg_negdstdst_negdst_neg_negunknown	unknown_0	unknown_1*
Tin
2	*/
Tout'
%2#*
_collective_manager_ids
 *�
_output_shapes�
�:���������:���������:@:���������@: :��������� :��������� :@ :���������@:@ :���������@: :��������� :��������� : :��������� :��������� :������������������ : :������������������ : :@ :������������������@:@ :������������������@: :������������������ :������������������ : :������������������ :������������������ :+��������������������������� : :+��������������������������� : *%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *
fR
__forward_call_16059�
$mean_squared_error/SquaredDifferenceSquaredDifference StatefulPartitionedCall:output:0labels*
T0*'
_output_shapes
:���������t
)mean_squared_error/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
����������
mean_squared_error/MeanMean(mean_squared_error/SquaredDifference:z:02mean_squared_error/Mean/reduction_indices:output:0*
T0*#
_output_shapes
:���������k
&mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
$mean_squared_error/weighted_loss/MulMul mean_squared_error/Mean:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*#
_output_shapes
:���������r
(mean_squared_error/weighted_loss/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
$mean_squared_error/weighted_loss/SumSum(mean_squared_error/weighted_loss/Mul:z:01mean_squared_error/weighted_loss/Const_1:output:0*
T0*
_output_shapes
: �
-mean_squared_error/weighted_loss/num_elementsSize(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
: �
2mean_squared_error/weighted_loss/num_elements/CastCast6mean_squared_error/weighted_loss/num_elements:output:0*

DstT0*

SrcT0*
_output_shapes
: g
%mean_squared_error/weighted_loss/RankConst*
_output_shapes
: *
dtype0*
value	B : n
,mean_squared_error/weighted_loss/range/startConst*
_output_shapes
: *
dtype0*
value	B : n
,mean_squared_error/weighted_loss/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :�
&mean_squared_error/weighted_loss/rangeRange5mean_squared_error/weighted_loss/range/start:output:0.mean_squared_error/weighted_loss/Rank:output:05mean_squared_error/weighted_loss/range/delta:output:0*
_output_shapes
: �
&mean_squared_error/weighted_loss/Sum_1Sum-mean_squared_error/weighted_loss/Sum:output:0/mean_squared_error/weighted_loss/range:output:0*
T0*
_output_shapes
: �
&mean_squared_error/weighted_loss/valueDivNoNan/mean_squared_error/weighted_loss/Sum_1:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: I
onesConst*
_output_shapes
: *
dtype0*
valueB
 *  �?}
:gradient_tape/mean_squared_error/weighted_loss/value/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
<gradient_tape/mean_squared_error/weighted_loss/value/Shape_1Const*
_output_shapes
: *
dtype0*
valueB �
Jgradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgsBroadcastGradientArgsCgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*2
_output_shapes 
:���������:����������
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNanones:output:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
8gradient_tape/mean_squared_error/weighted_loss/value/SumSumCgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
: �
<gradient_tape/mean_squared_error/weighted_loss/value/ReshapeReshapeAgradient_tape/mean_squared_error/weighted_loss/value/Sum:output:0Cgradient_tape/mean_squared_error/weighted_loss/value/Shape:output:0*
T0*
_output_shapes
: �
8gradient_tape/mean_squared_error/weighted_loss/value/NegNeg/mean_squared_error/weighted_loss/Sum_1:output:0*
T0*
_output_shapes
: �
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1DivNoNan<gradient_tape/mean_squared_error/weighted_loss/value/Neg:y:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
Agradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2DivNoNanEgradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_1:z:06mean_squared_error/weighted_loss/num_elements/Cast:y:0*
T0*
_output_shapes
: �
8gradient_tape/mean_squared_error/weighted_loss/value/mulMulones:output:0Egradient_tape/mean_squared_error/weighted_loss/value/div_no_nan_2:z:0*
T0*
_output_shapes
: �
:gradient_tape/mean_squared_error/weighted_loss/value/Sum_1Sum<gradient_tape/mean_squared_error/weighted_loss/value/mul:z:0Ogradient_tape/mean_squared_error/weighted_loss/value/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
: �
>gradient_tape/mean_squared_error/weighted_loss/value/Reshape_1ReshapeCgradient_tape/mean_squared_error/weighted_loss/value/Sum_1:output:0Egradient_tape/mean_squared_error/weighted_loss/value/Shape_1:output:0*
T0*
_output_shapes
: 
<gradient_tape/mean_squared_error/weighted_loss/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB �
>gradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB �
6gradient_tape/mean_squared_error/weighted_loss/ReshapeReshapeEgradient_tape/mean_squared_error/weighted_loss/value/Reshape:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape/shape_1:output:0*
T0*
_output_shapes
: w
4gradient_tape/mean_squared_error/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB �
3gradient_tape/mean_squared_error/weighted_loss/TileTile?gradient_tape/mean_squared_error/weighted_loss/Reshape:output:0=gradient_tape/mean_squared_error/weighted_loss/Const:output:0*
T0*
_output_shapes
: �
>gradient_tape/mean_squared_error/weighted_loss/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:�
8gradient_tape/mean_squared_error/weighted_loss/Reshape_1Reshape<gradient_tape/mean_squared_error/weighted_loss/Tile:output:0Ggradient_tape/mean_squared_error/weighted_loss/Reshape_1/shape:output:0*
T0*
_output_shapes
:�
4gradient_tape/mean_squared_error/weighted_loss/ShapeShape(mean_squared_error/weighted_loss/Mul:z:0*
T0*
_output_shapes
:�
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileAgradient_tape/mean_squared_error/weighted_loss/Reshape_1:output:0=gradient_tape/mean_squared_error/weighted_loss/Shape:output:0*
T0*#
_output_shapes
:����������
2gradient_tape/mean_squared_error/weighted_loss/MulMul>gradient_tape/mean_squared_error/weighted_loss/Tile_1:output:0/mean_squared_error/weighted_loss/Const:output:0*
T0*#
_output_shapes
:���������~
&gradient_tape/mean_squared_error/ShapeShape(mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:�
%gradient_tape/mean_squared_error/SizeConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B :�
$gradient_tape/mean_squared_error/addAddV22mean_squared_error/Mean/reduction_indices:output:0.gradient_tape/mean_squared_error/Size:output:0*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: �
$gradient_tape/mean_squared_error/modFloorMod(gradient_tape/mean_squared_error/add:z:0.gradient_tape/mean_squared_error/Size:output:0*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: �
(gradient_tape/mean_squared_error/Shape_1Const*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
valueB �
,gradient_tape/mean_squared_error/range/startConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B : �
,gradient_tape/mean_squared_error/range/deltaConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B :�
&gradient_tape/mean_squared_error/rangeRange5gradient_tape/mean_squared_error/range/start:output:0.gradient_tape/mean_squared_error/Size:output:05gradient_tape/mean_squared_error/range/delta:output:0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
:�
+gradient_tape/mean_squared_error/ones/ConstConst*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: *
dtype0*
value	B :�
%gradient_tape/mean_squared_error/onesFill1gradient_tape/mean_squared_error/Shape_1:output:04gradient_tape/mean_squared_error/ones/Const:output:0*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
: �
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch/gradient_tape/mean_squared_error/range:output:0(gradient_tape/mean_squared_error/mod:z:0/gradient_tape/mean_squared_error/Shape:output:0.gradient_tape/mean_squared_error/ones:output:0*
N*
T0*9
_class/
-+loc:@gradient_tape/mean_squared_error/Shape*
_output_shapes
:�
(gradient_tape/mean_squared_error/ReshapeReshape6gradient_tape/mean_squared_error/weighted_loss/Mul:z:07gradient_tape/mean_squared_error/DynamicStitch:merged:0*
T0*0
_output_shapes
:�������������������
,gradient_tape/mean_squared_error/BroadcastToBroadcastTo1gradient_tape/mean_squared_error/Reshape:output:0/gradient_tape/mean_squared_error/Shape:output:0*
T0*'
_output_shapes
:����������
(gradient_tape/mean_squared_error/Shape_2Shape(mean_squared_error/SquaredDifference:z:0*
T0*
_output_shapes
:x
(gradient_tape/mean_squared_error/Shape_3Shape mean_squared_error/Mean:output:0*
T0*
_output_shapes
:p
&gradient_tape/mean_squared_error/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
%gradient_tape/mean_squared_error/ProdProd1gradient_tape/mean_squared_error/Shape_2:output:0/gradient_tape/mean_squared_error/Const:output:0*
T0*
_output_shapes
: r
(gradient_tape/mean_squared_error/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
'gradient_tape/mean_squared_error/Prod_1Prod1gradient_tape/mean_squared_error/Shape_3:output:01gradient_tape/mean_squared_error/Const_1:output:0*
T0*
_output_shapes
: l
*gradient_tape/mean_squared_error/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
(gradient_tape/mean_squared_error/MaximumMaximum0gradient_tape/mean_squared_error/Prod_1:output:03gradient_tape/mean_squared_error/Maximum/y:output:0*
T0*
_output_shapes
: �
)gradient_tape/mean_squared_error/floordivFloorDiv.gradient_tape/mean_squared_error/Prod:output:0,gradient_tape/mean_squared_error/Maximum:z:0*
T0*
_output_shapes
: �
%gradient_tape/mean_squared_error/CastCast-gradient_tape/mean_squared_error/floordiv:z:0*

DstT0*

SrcT0*
_output_shapes
: �
(gradient_tape/mean_squared_error/truedivRealDiv5gradient_tape/mean_squared_error/BroadcastTo:output:0)gradient_tape/mean_squared_error/Cast:y:0*
T0*'
_output_shapes
:����������
'gradient_tape/mean_squared_error/scalarConst)^gradient_tape/mean_squared_error/truediv*
_output_shapes
: *
dtype0*
valueB
 *   @�
$gradient_tape/mean_squared_error/MulMul0gradient_tape/mean_squared_error/scalar:output:0,gradient_tape/mean_squared_error/truediv:z:0*
T0*'
_output_shapes
:����������
$gradient_tape/mean_squared_error/subSub StatefulPartitionedCall:output:0labels)^gradient_tape/mean_squared_error/truediv*
T0*'
_output_shapes
:����������
&gradient_tape/mean_squared_error/mul_1Mul(gradient_tape/mean_squared_error/Mul:z:0(gradient_tape/mean_squared_error/sub:z:0*
T0*'
_output_shapes
:���������x
(gradient_tape/mean_squared_error/Shape_4Shape StatefulPartitionedCall:output:0*
T0*
_output_shapes
:^
(gradient_tape/mean_squared_error/Shape_5Shapelabels*
T0*
_output_shapes
:�
6gradient_tape/mean_squared_error/BroadcastGradientArgsBroadcastGradientArgs1gradient_tape/mean_squared_error/Shape_4:output:01gradient_tape/mean_squared_error/Shape_5:output:0*2
_output_shapes 
:���������:����������
$gradient_tape/mean_squared_error/SumSum*gradient_tape/mean_squared_error/mul_1:z:0;gradient_tape/mean_squared_error/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
:�
*gradient_tape/mean_squared_error/Reshape_1Reshape-gradient_tape/mean_squared_error/Sum:output:01gradient_tape/mean_squared_error/Shape_4:output:0*
T0*'
_output_shapes
:����������
PartitionedCallPartitionedCall3gradient_tape/mean_squared_error/Reshape_1:output:0 StatefulPartitionedCall:output:1 StatefulPartitionedCall:output:2 StatefulPartitionedCall:output:3 StatefulPartitionedCall:output:4 StatefulPartitionedCall:output:5 StatefulPartitionedCall:output:6 StatefulPartitionedCall:output:7 StatefulPartitionedCall:output:8 StatefulPartitionedCall:output:9!StatefulPartitionedCall:output:10!StatefulPartitionedCall:output:11!StatefulPartitionedCall:output:12!StatefulPartitionedCall:output:13!StatefulPartitionedCall:output:14!StatefulPartitionedCall:output:15!StatefulPartitionedCall:output:16!StatefulPartitionedCall:output:17!StatefulPartitionedCall:output:18!StatefulPartitionedCall:output:19!StatefulPartitionedCall:output:20!StatefulPartitionedCall:output:21!StatefulPartitionedCall:output:22!StatefulPartitionedCall:output:23!StatefulPartitionedCall:output:24!StatefulPartitionedCall:output:25!StatefulPartitionedCall:output:26!StatefulPartitionedCall:output:27!StatefulPartitionedCall:output:28!StatefulPartitionedCall:output:29!StatefulPartitionedCall:output:30!StatefulPartitionedCall:output:31!StatefulPartitionedCall:output:32!StatefulPartitionedCall:output:33!StatefulPartitionedCall:output:34*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *�
_output_shapes�
�:��������� :������������������ :+��������������������������� :��������� :������������������ :+��������������������������� :@ :@ :@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *0
f+R)
'__inference___backward_call_15826_16060p
SGD/Cast/ReadVariableOpReadVariableOp sgd_cast_readvariableop_resource*
_output_shapes
: *
dtype0�
SGD/IdentityIdentitySGD/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*
_output_shapes
: t
SGD/Cast_1/ReadVariableOpReadVariableOp"sgd_cast_1_readvariableop_resource*
_output_shapes
: *
dtype0�
SGD/Identity_1Identity!SGD/Cast_1/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*
_output_shapes
: ]
SGD/Identity_2IdentityPartitionedCall:output:7*
T0*
_output_shapes

:@ ]
SGD/Identity_3IdentityPartitionedCall:output:6*
T0*
_output_shapes

:@ ]
SGD/Identity_4IdentityPartitionedCall:output:8*
T0*
_output_shapes

:@�
SGD/IdentityN	IdentityNPartitionedCall:output:7PartitionedCall:output:6PartitionedCall:output:8PartitionedCall:output:7PartitionedCall:output:6PartitionedCall:output:8*
T

2*+
_gradient_op_typeCustomGradient-16110*P
_output_shapes>
<:@ :@ :@:@ :@ :@�
+SGD/SGD/update/ResourceApplyGradientDescentResourceApplyGradientDescent	unknown_0SGD/Identity:output:0SGD/IdentityN:output:0^StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:GPU:0*
T0*
_class

loc:@15748*
_output_shapes
 *
use_locking(�
-SGD/SGD/update_1/ResourceApplyGradientDescentResourceApplyGradientDescentunknownSGD/Identity:output:0SGD/IdentityN:output:1^StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:GPU:0*
T0*
_class

loc:@15746*
_output_shapes
 *
use_locking(�
-SGD/SGD/update_2/ResourceApplyGradientDescentResourceApplyGradientDescent	unknown_1SGD/Identity:output:0SGD/IdentityN:output:2^StatefulPartitionedCall",/job:localhost/replica:0/task:0/device:GPU:0*
T0*
_class

loc:@15750*
_output_shapes
 *
use_locking(�
SGD/SGD/group_depsNoOp,^SGD/SGD/update/ResourceApplyGradientDescent.^SGD/SGD/update_1/ResourceApplyGradientDescent.^SGD/SGD/update_2/ResourceApplyGradientDescent",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 d
SGD/SGD/ConstConst^SGD/SGD/group_deps*
_output_shapes
: *
dtype0	*
value	B	 R�
SGD/SGD/AssignAddVariableOpAssignAddVariableOp$sgd_sgd_assignaddvariableop_resourceSGD/SGD/Const:output:0*
_output_shapes
 *
dtype0	h
IdentityIdentity*mean_squared_error/weighted_loss/value:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^SGD/Cast/ReadVariableOp^SGD/Cast_1/ReadVariableOp^SGD/SGD/AssignAddVariableOp,^SGD/SGD/update/ResourceApplyGradientDescent.^SGD/SGD/update_1/ResourceApplyGradientDescent.^SGD/SGD/update_2/ResourceApplyGradientDescent^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :������������������ :+��������������������������� :��������� :������������������ :+��������������������������� :���������: : : : : : 22
SGD/Cast/ReadVariableOpSGD/Cast/ReadVariableOp26
SGD/Cast_1/ReadVariableOpSGD/Cast_1/ReadVariableOp2:
SGD/SGD/AssignAddVariableOpSGD/SGD/AssignAddVariableOp2Z
+SGD/SGD/update/ResourceApplyGradientDescent+SGD/SGD/update/ResourceApplyGradientDescent2^
-SGD/SGD/update_1/ResourceApplyGradientDescent-SGD/SGD/update_1/ResourceApplyGradientDescent2^
-SGD/SGD/update_2/ResourceApplyGradientDescent-SGD/SGD/update_2/ResourceApplyGradientDescent22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:��������� 

_user_specified_namesrc:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	src_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namesrc_neg_neg:LH
'
_output_shapes
:��������� 

_user_specified_namedst:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	dst_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namedst_neg_neg:OK
'
_output_shapes
:���������
 
_user_specified_namelabels
�(
�
__inference_call_15718
src
src_neg
src_neg_neg
dst
dst_neg
dst_neg_neg0
matmul_readvariableop_resource:@ 2
 matmul_2_readvariableop_resource:@ 6
$dense_matmul_readvariableop_resource:@
identity��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�MatMul_3/ReadVariableOp�dense/MatMul/ReadVariableOpX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :y
MeanMeandst_neg_negMean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������ Z
Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :}
Mean_1Meansrc_neg_neg!Mean_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������ M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2dst_negMean:output:0concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������@O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_1ConcatV2src_negMean_1:output:0concat_1/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������@t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
MatMulBatchMatMulV2concat:output:0MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ v
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
MatMul_1BatchMatMulV2concat_1:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ Z
Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
Mean_2MeanMatMul:output:0!Mean_2/reduction_indices:output:0*
T0*'
_output_shapes
:��������� Z
Mean_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
Mean_3MeanMatMul_1:output:0!Mean_3/reduction_indices:output:0*
T0*'
_output_shapes
:��������� O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concat_2ConcatV2dstMean_2:output:0concat_2/axis:output:0*
N*
T0*'
_output_shapes
:���������@O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :}
concat_3ConcatV2srcMean_3:output:0concat_3/axis:output:0*
N*
T0*'
_output_shapes
:���������@x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@ *
dtype0x
MatMul_2MatMulconcat_2:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_3/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@ *
dtype0x
MatMul_3MatMulconcat_3:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concat_4ConcatV2MatMul_3:product:0MatMul_2:product:0concat_4/axis:output:0*
N*
T0*'
_output_shapes
:���������@�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/MatMulMatMulconcat_4:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������\

dense/ReluReludense/MatMul:product:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :������������������ :+��������������������������� :��������� :������������������ :+��������������������������� : : : 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:L H
'
_output_shapes
:��������� 

_user_specified_namesrc:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	src_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namesrc_neg_neg:LH
'
_output_shapes
:��������� 

_user_specified_namedst:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	dst_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namedst_neg_neg
�
�
#__inference_signature_wrapper_15736
dst
dst_neg
dst_neg_neg
src
src_neg
src_neg_neg
unknown:@ 
	unknown_0:@ 
	unknown_1:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallsrcsrc_negsrc_neg_negdstdst_negdst_neg_negunknown	unknown_0	unknown_1*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *
fR
__inference_call_15718o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :������������������ :+��������������������������� :��������� :������������������ :+��������������������������� : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:��������� 

_user_specified_namedst:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	dst_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namedst_neg_neg:LH
'
_output_shapes
:��������� 

_user_specified_namesrc:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	src_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namesrc_neg_neg
�
�
__inference__traced_save_16197
file_prefix+
'savev2_dense_kernel_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop>
:savev2_mean_aggregation_layer_1_weight_read_readvariableop>
:savev2_mean_aggregation_layer_2_weight_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBGseq_layers/0/mean_aggregation_layer_1_weight/.ATTRIBUTES/VARIABLE_VALUEBGseq_layers/1/mean_aggregation_layer_2_weight/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH}
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop:savev2_mean_aggregation_layer_1_weight_read_readvariableop:savev2_mean_aggregation_layer_2_weight_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes

2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*=
_input_shapes,
*: :@: : : : :@ :@ : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@ :$ 

_output_shapes

:@ :

_output_shapes
: 
�
�
"__inference_internal_grad_fn_16171
result_grads_0
result_grads_1
result_grads_2
result_grads_3
result_grads_4
result_grads_5

identity_3

identity_4

identity_5M
IdentityIdentityresult_grads_0*
T0*
_output_shapes

:@ O

Identity_1Identityresult_grads_1*
T0*
_output_shapes

:@ O

Identity_2Identityresult_grads_2*
T0*
_output_shapes

:@�
	IdentityN	IdentityNresult_grads_0result_grads_1result_grads_2result_grads_0result_grads_1result_grads_2*
T

2*+
_gradient_op_typeCustomGradient-16158*P
_output_shapes>
<:@ :@ :@:@ :@ :@S

Identity_3IdentityIdentityN:output:0*
T0*
_output_shapes

:@ S

Identity_4IdentityIdentityN:output:1*
T0*
_output_shapes

:@ S

Identity_5IdentityIdentityN:output:2*
T0*
_output_shapes

:@"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*O
_input_shapes>
<:@ :@ :@:@ :@ :@:N J

_output_shapes

:@ 
(
_user_specified_nameresult_grads_0:NJ

_output_shapes

:@ 
(
_user_specified_nameresult_grads_1:NJ

_output_shapes

:@
(
_user_specified_nameresult_grads_2:NJ

_output_shapes

:@ 
(
_user_specified_nameresult_grads_3:NJ

_output_shapes

:@ 
(
_user_specified_nameresult_grads_4:NJ

_output_shapes

:@
(
_user_specified_nameresult_grads_5
��
�
'__inference___backward_call_15826_16060
placeholder1
-gradients_dense_relu_grad_relugrad_dense_reluB
>gradients_dense_matmul_grad_matmul_dense_matmul_readvariableop1
-gradients_dense_matmul_grad_matmul_1_concat_4-
)gradients_concat_4_grad_mod_concat_4_axis*
&gradients_concat_4_grad_shape_matmul_3+
'gradients_concat_4_grad_shapen_matmul_2:
6gradients_matmul_3_grad_matmul_matmul_3_readvariableop-
)gradients_matmul_3_grad_matmul_1_concat_3:
6gradients_matmul_2_grad_matmul_matmul_2_readvariableop-
)gradients_matmul_2_grad_matmul_1_concat_2-
)gradients_concat_3_grad_mod_concat_3_axis%
!gradients_concat_3_grad_shape_src)
%gradients_concat_3_grad_shapen_mean_3-
)gradients_concat_2_grad_mod_concat_2_axis%
!gradients_concat_2_grad_shape_dst)
%gradients_concat_2_grad_shapen_mean_2(
$gradients_mean_3_grad_shape_matmul_16
2gradients_mean_3_grad_add_mean_3_reduction_indices&
"gradients_mean_2_grad_shape_matmul6
2gradients_mean_2_grad_add_mean_2_reduction_indices:
6gradients_matmul_1_grad_matmul_matmul_1_readvariableop-
)gradients_matmul_1_grad_matmul_1_concat_16
2gradients_matmul_grad_matmul_matmul_readvariableop)
%gradients_matmul_grad_matmul_1_concat-
)gradients_concat_1_grad_mod_concat_1_axis)
%gradients_concat_1_grad_shape_src_neg)
%gradients_concat_1_grad_shapen_mean_1)
%gradients_concat_grad_mod_concat_axis'
#gradients_concat_grad_shape_dst_neg%
!gradients_concat_grad_shapen_mean+
'gradients_mean_1_grad_shape_src_neg_neg6
2gradients_mean_1_grad_add_mean_1_reduction_indices)
%gradients_mean_grad_shape_dst_neg_neg2
.gradients_mean_grad_add_mean_reduction_indices
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:����������
"gradients/dense/Relu_grad/ReluGradReluGradgradients/grad_ys_0:output:0-gradients_dense_relu_grad_relugrad_dense_relu*
T0*'
_output_shapes
:����������
"gradients/dense/MatMul_grad/MatMulMatMul.gradients/dense/Relu_grad/ReluGrad:backprops:0>gradients_dense_matmul_grad_matmul_dense_matmul_readvariableop*
T0*'
_output_shapes
:���������@*
transpose_b(�
$gradients/dense/MatMul_grad/MatMul_1MatMul-gradients_dense_matmul_grad_matmul_1_concat_4.gradients/dense/Relu_grad/ReluGrad:backprops:0*
T0*
_output_shapes

:@*
transpose_a(^
gradients/concat_4_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_4_grad/modFloorMod)gradients_concat_4_grad_mod_concat_4_axis%gradients/concat_4_grad/Rank:output:0*
T0*
_output_shapes
: s
gradients/concat_4_grad/ShapeShape&gradients_concat_4_grad_shape_matmul_3*
T0*
_output_shapes
:�
gradients/concat_4_grad/ShapeNShapeN&gradients_concat_4_grad_shape_matmul_3'gradients_concat_4_grad_shapen_matmul_2*
N*
T0* 
_output_shapes
::�
$gradients/concat_4_grad/ConcatOffsetConcatOffsetgradients/concat_4_grad/mod:z:0'gradients/concat_4_grad/ShapeN:output:0'gradients/concat_4_grad/ShapeN:output:1*
N* 
_output_shapes
::�
gradients/concat_4_grad/SliceSlice,gradients/dense/MatMul_grad/MatMul:product:0-gradients/concat_4_grad/ConcatOffset:offset:0'gradients/concat_4_grad/ShapeN:output:0*
Index0*
T0*'
_output_shapes
:��������� �
gradients/concat_4_grad/Slice_1Slice,gradients/dense/MatMul_grad/MatMul:product:0-gradients/concat_4_grad/ConcatOffset:offset:1'gradients/concat_4_grad/ShapeN:output:1*
Index0*
T0*'
_output_shapes
:��������� �
gradients/MatMul_3_grad/MatMulMatMul&gradients/concat_4_grad/Slice:output:06gradients_matmul_3_grad_matmul_matmul_3_readvariableop*
T0*'
_output_shapes
:���������@*
transpose_b(�
 gradients/MatMul_3_grad/MatMul_1MatMul)gradients_matmul_3_grad_matmul_1_concat_3&gradients/concat_4_grad/Slice:output:0*
T0*
_output_shapes

:@ *
transpose_a(�
gradients/MatMul_2_grad/MatMulMatMul(gradients/concat_4_grad/Slice_1:output:06gradients_matmul_2_grad_matmul_matmul_2_readvariableop*
T0*'
_output_shapes
:���������@*
transpose_b(�
 gradients/MatMul_2_grad/MatMul_1MatMul)gradients_matmul_2_grad_matmul_1_concat_2(gradients/concat_4_grad/Slice_1:output:0*
T0*
_output_shapes

:@ *
transpose_a(^
gradients/concat_3_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_3_grad/modFloorMod)gradients_concat_3_grad_mod_concat_3_axis%gradients/concat_3_grad/Rank:output:0*
T0*
_output_shapes
: n
gradients/concat_3_grad/ShapeShape!gradients_concat_3_grad_shape_src*
T0*
_output_shapes
:�
gradients/concat_3_grad/ShapeNShapeN!gradients_concat_3_grad_shape_src%gradients_concat_3_grad_shapen_mean_3*
N*
T0* 
_output_shapes
::�
$gradients/concat_3_grad/ConcatOffsetConcatOffsetgradients/concat_3_grad/mod:z:0'gradients/concat_3_grad/ShapeN:output:0'gradients/concat_3_grad/ShapeN:output:1*
N* 
_output_shapes
::�
gradients/concat_3_grad/SliceSlice(gradients/MatMul_3_grad/MatMul:product:0-gradients/concat_3_grad/ConcatOffset:offset:0'gradients/concat_3_grad/ShapeN:output:0*
Index0*
T0*'
_output_shapes
:��������� �
gradients/concat_3_grad/Slice_1Slice(gradients/MatMul_3_grad/MatMul:product:0-gradients/concat_3_grad/ConcatOffset:offset:1'gradients/concat_3_grad/ShapeN:output:1*
Index0*
T0*'
_output_shapes
:��������� ^
gradients/concat_2_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_2_grad/modFloorMod)gradients_concat_2_grad_mod_concat_2_axis%gradients/concat_2_grad/Rank:output:0*
T0*
_output_shapes
: n
gradients/concat_2_grad/ShapeShape!gradients_concat_2_grad_shape_dst*
T0*
_output_shapes
:�
gradients/concat_2_grad/ShapeNShapeN!gradients_concat_2_grad_shape_dst%gradients_concat_2_grad_shapen_mean_2*
N*
T0* 
_output_shapes
::�
$gradients/concat_2_grad/ConcatOffsetConcatOffsetgradients/concat_2_grad/mod:z:0'gradients/concat_2_grad/ShapeN:output:0'gradients/concat_2_grad/ShapeN:output:1*
N* 
_output_shapes
::�
gradients/concat_2_grad/SliceSlice(gradients/MatMul_2_grad/MatMul:product:0-gradients/concat_2_grad/ConcatOffset:offset:0'gradients/concat_2_grad/ShapeN:output:0*
Index0*
T0*'
_output_shapes
:��������� �
gradients/concat_2_grad/Slice_1Slice(gradients/MatMul_2_grad/MatMul:product:0-gradients/concat_2_grad/ConcatOffset:offset:1'gradients/concat_2_grad/ShapeN:output:1*
Index0*
T0*'
_output_shapes
:��������� o
gradients/Mean_3_grad/ShapeShape$gradients_mean_3_grad_shape_matmul_1*
T0*
_output_shapes
:�
gradients/Mean_3_grad/SizeConst*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_3_grad/addAddV22gradients_mean_3_grad_add_mean_3_reduction_indices#gradients/Mean_3_grad/Size:output:0*
T0*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: �
gradients/Mean_3_grad/modFloorModgradients/Mean_3_grad/add:z:0#gradients/Mean_3_grad/Size:output:0*
T0*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: �
gradients/Mean_3_grad/Shape_1Const*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: *
dtype0*
valueB �
!gradients/Mean_3_grad/range/startConst*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: *
dtype0*
value	B : �
!gradients/Mean_3_grad/range/deltaConst*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_3_grad/rangeRange*gradients/Mean_3_grad/range/start:output:0#gradients/Mean_3_grad/Size:output:0*gradients/Mean_3_grad/range/delta:output:0*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
:�
 gradients/Mean_3_grad/ones/ConstConst*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_3_grad/onesFill&gradients/Mean_3_grad/Shape_1:output:0)gradients/Mean_3_grad/ones/Const:output:0*
T0*.
_class$
" loc:@gradients/Mean_3_grad/Shape*
_output_shapes
: �
#gradients/Mean_3_grad/DynamicStitchDynamicStitch$gradients/Mean_3_grad/range:output:0gradients/Mean_3_grad/mod:z:0$gradients/Mean_3_grad/Shape:output:0#gradients/Mean_3_grad/ones:output:0*
N*
T0*.
_class$
" loc:@gradients/Mean_3_grad/Shape*#
_output_shapes
:����������
gradients/Mean_3_grad/ReshapeReshape(gradients/concat_3_grad/Slice_1:output:0,gradients/Mean_3_grad/DynamicStitch:merged:0*
T0*
_output_shapes
:�
!gradients/Mean_3_grad/BroadcastToBroadcastTo&gradients/Mean_3_grad/Reshape:output:0$gradients/Mean_3_grad/Shape:output:0*
T0*4
_output_shapes"
 :������������������ q
gradients/Mean_3_grad/Shape_2Shape$gradients_mean_3_grad_shape_matmul_1*
T0*
_output_shapes
:r
gradients/Mean_3_grad/Shape_3Shape%gradients_concat_3_grad_shapen_mean_3*
T0*
_output_shapes
:e
gradients/Mean_3_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_3_grad/ProdProd&gradients/Mean_3_grad/Shape_2:output:0$gradients/Mean_3_grad/Const:output:0*
T0*
_output_shapes
: g
gradients/Mean_3_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_3_grad/Prod_1Prod&gradients/Mean_3_grad/Shape_3:output:0&gradients/Mean_3_grad/Const_1:output:0*
T0*
_output_shapes
: a
gradients/Mean_3_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_3_grad/MaximumMaximum%gradients/Mean_3_grad/Prod_1:output:0(gradients/Mean_3_grad/Maximum/y:output:0*
T0*
_output_shapes
: �
gradients/Mean_3_grad/floordivFloorDiv#gradients/Mean_3_grad/Prod:output:0!gradients/Mean_3_grad/Maximum:z:0*
T0*
_output_shapes
: v
gradients/Mean_3_grad/CastCast"gradients/Mean_3_grad/floordiv:z:0*

DstT0*

SrcT0*
_output_shapes
: �
gradients/Mean_3_grad/truedivRealDiv*gradients/Mean_3_grad/BroadcastTo:output:0gradients/Mean_3_grad/Cast:y:0*
T0*4
_output_shapes"
 :������������������ m
gradients/Mean_2_grad/ShapeShape"gradients_mean_2_grad_shape_matmul*
T0*
_output_shapes
:�
gradients/Mean_2_grad/SizeConst*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_2_grad/addAddV22gradients_mean_2_grad_add_mean_2_reduction_indices#gradients/Mean_2_grad/Size:output:0*
T0*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: �
gradients/Mean_2_grad/modFloorModgradients/Mean_2_grad/add:z:0#gradients/Mean_2_grad/Size:output:0*
T0*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: �
gradients/Mean_2_grad/Shape_1Const*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: *
dtype0*
valueB �
!gradients/Mean_2_grad/range/startConst*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B : �
!gradients/Mean_2_grad/range/deltaConst*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_2_grad/rangeRange*gradients/Mean_2_grad/range/start:output:0#gradients/Mean_2_grad/Size:output:0*gradients/Mean_2_grad/range/delta:output:0*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
:�
 gradients/Mean_2_grad/ones/ConstConst*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_2_grad/onesFill&gradients/Mean_2_grad/Shape_1:output:0)gradients/Mean_2_grad/ones/Const:output:0*
T0*.
_class$
" loc:@gradients/Mean_2_grad/Shape*
_output_shapes
: �
#gradients/Mean_2_grad/DynamicStitchDynamicStitch$gradients/Mean_2_grad/range:output:0gradients/Mean_2_grad/mod:z:0$gradients/Mean_2_grad/Shape:output:0#gradients/Mean_2_grad/ones:output:0*
N*
T0*.
_class$
" loc:@gradients/Mean_2_grad/Shape*#
_output_shapes
:����������
gradients/Mean_2_grad/ReshapeReshape(gradients/concat_2_grad/Slice_1:output:0,gradients/Mean_2_grad/DynamicStitch:merged:0*
T0*
_output_shapes
:�
!gradients/Mean_2_grad/BroadcastToBroadcastTo&gradients/Mean_2_grad/Reshape:output:0$gradients/Mean_2_grad/Shape:output:0*
T0*4
_output_shapes"
 :������������������ o
gradients/Mean_2_grad/Shape_2Shape"gradients_mean_2_grad_shape_matmul*
T0*
_output_shapes
:r
gradients/Mean_2_grad/Shape_3Shape%gradients_concat_2_grad_shapen_mean_2*
T0*
_output_shapes
:e
gradients/Mean_2_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_2_grad/ProdProd&gradients/Mean_2_grad/Shape_2:output:0$gradients/Mean_2_grad/Const:output:0*
T0*
_output_shapes
: g
gradients/Mean_2_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_2_grad/Prod_1Prod&gradients/Mean_2_grad/Shape_3:output:0&gradients/Mean_2_grad/Const_1:output:0*
T0*
_output_shapes
: a
gradients/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_2_grad/MaximumMaximum%gradients/Mean_2_grad/Prod_1:output:0(gradients/Mean_2_grad/Maximum/y:output:0*
T0*
_output_shapes
: �
gradients/Mean_2_grad/floordivFloorDiv#gradients/Mean_2_grad/Prod:output:0!gradients/Mean_2_grad/Maximum:z:0*
T0*
_output_shapes
: v
gradients/Mean_2_grad/CastCast"gradients/Mean_2_grad/floordiv:z:0*

DstT0*

SrcT0*
_output_shapes
: �
gradients/Mean_2_grad/truedivRealDiv*gradients/Mean_2_grad/BroadcastTo:output:0gradients/Mean_2_grad/Cast:y:0*
T0*4
_output_shapes"
 :������������������ �
gradients/AddNAddN*gradients/MatMul_3_grad/MatMul_1:product:0*gradients/MatMul_2_grad/MatMul_1:product:0*
N*
T0*3
_class)
'%loc:@gradients/MatMul_3_grad/MatMul_1*
_output_shapes

:@ �
gradients/MatMul_1_grad/MatMulBatchMatMulV2!gradients/Mean_3_grad/truediv:z:06gradients_matmul_1_grad_matmul_matmul_1_readvariableop*
T0*4
_output_shapes"
 :������������������@*
adj_y(�
 gradients/MatMul_1_grad/MatMul_1BatchMatMulV2)gradients_matmul_1_grad_matmul_1_concat_1!gradients/Mean_3_grad/truediv:z:0*
T0*+
_output_shapes
:���������@ *
adj_x(v
gradients/MatMul_1_grad/ShapeShape)gradients_matmul_1_grad_matmul_1_concat_1*
T0*
_output_shapes
:p
gradients/MatMul_1_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@       u
+gradients/MatMul_1_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: �
-gradients/MatMul_1_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������w
-gradients/MatMul_1_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%gradients/MatMul_1_grad/strided_sliceStridedSlice&gradients/MatMul_1_grad/Shape:output:04gradients/MatMul_1_grad/strided_slice/stack:output:06gradients/MatMul_1_grad/strided_slice/stack_1:output:06gradients/MatMul_1_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-gradients/MatMul_1_grad/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
/gradients/MatMul_1_grad/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������y
/gradients/MatMul_1_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
'gradients/MatMul_1_grad/strided_slice_1StridedSlice(gradients/MatMul_1_grad/Shape_1:output:06gradients/MatMul_1_grad/strided_slice_1/stack:output:08gradients/MatMul_1_grad/strided_slice_1/stack_1:output:08gradients/MatMul_1_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask�
-gradients/MatMul_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients/MatMul_1_grad/strided_slice:output:00gradients/MatMul_1_grad/strided_slice_1:output:0*2
_output_shapes 
:���������:����������
gradients/MatMul_1_grad/SumSum'gradients/MatMul_1_grad/MatMul:output:02gradients/MatMul_1_grad/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
:�
gradients/MatMul_1_grad/ReshapeReshape$gradients/MatMul_1_grad/Sum:output:0&gradients/MatMul_1_grad/Shape:output:0*
T0*4
_output_shapes"
 :������������������@�
gradients/MatMul_1_grad/Sum_1Sum)gradients/MatMul_1_grad/MatMul_1:output:02gradients/MatMul_1_grad/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
:�
!gradients/MatMul_1_grad/Reshape_1Reshape&gradients/MatMul_1_grad/Sum_1:output:0(gradients/MatMul_1_grad/Shape_1:output:0*
T0*
_output_shapes

:@ �
gradients/MatMul_grad/MatMulBatchMatMulV2!gradients/Mean_2_grad/truediv:z:02gradients_matmul_grad_matmul_matmul_readvariableop*
T0*4
_output_shapes"
 :������������������@*
adj_y(�
gradients/MatMul_grad/MatMul_1BatchMatMulV2%gradients_matmul_grad_matmul_1_concat!gradients/Mean_2_grad/truediv:z:0*
T0*+
_output_shapes
:���������@ *
adj_x(p
gradients/MatMul_grad/ShapeShape%gradients_matmul_grad_matmul_1_concat*
T0*
_output_shapes
:n
gradients/MatMul_grad/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"@       s
)gradients/MatMul_grad/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
+gradients/MatMul_grad/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������u
+gradients/MatMul_grad/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#gradients/MatMul_grad/strided_sliceStridedSlice$gradients/MatMul_grad/Shape:output:02gradients/MatMul_grad/strided_slice/stack:output:04gradients/MatMul_grad/strided_slice/stack_1:output:04gradients/MatMul_grad/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+gradients/MatMul_grad/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: �
-gradients/MatMul_grad/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
���������w
-gradients/MatMul_grad/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%gradients/MatMul_grad/strided_slice_1StridedSlice&gradients/MatMul_grad/Shape_1:output:04gradients/MatMul_grad/strided_slice_1/stack:output:06gradients/MatMul_grad/strided_slice_1/stack_1:output:06gradients/MatMul_grad/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask�
+gradients/MatMul_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/MatMul_grad/strided_slice:output:0.gradients/MatMul_grad/strided_slice_1:output:0*2
_output_shapes 
:���������:����������
gradients/MatMul_grad/SumSum%gradients/MatMul_grad/MatMul:output:00gradients/MatMul_grad/BroadcastGradientArgs:r0:0*
T0*
_output_shapes
:�
gradients/MatMul_grad/ReshapeReshape"gradients/MatMul_grad/Sum:output:0$gradients/MatMul_grad/Shape:output:0*
T0*4
_output_shapes"
 :������������������@�
gradients/MatMul_grad/Sum_1Sum'gradients/MatMul_grad/MatMul_1:output:00gradients/MatMul_grad/BroadcastGradientArgs:r1:0*
T0*
_output_shapes
:�
gradients/MatMul_grad/Reshape_1Reshape$gradients/MatMul_grad/Sum_1:output:0&gradients/MatMul_grad/Shape_1:output:0*
T0*
_output_shapes

:@ ^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: r
gradients/concat_1_grad/ShapeShape%gradients_concat_1_grad_shape_src_neg*
T0*
_output_shapes
:�
gradients/concat_1_grad/ShapeNShapeN%gradients_concat_1_grad_shape_src_neg%gradients_concat_1_grad_shapen_mean_1*
N*
T0* 
_output_shapes
::�
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0'gradients/concat_1_grad/ShapeN:output:0'gradients/concat_1_grad/ShapeN:output:1*
N* 
_output_shapes
::�
gradients/concat_1_grad/SliceSlice(gradients/MatMul_1_grad/Reshape:output:0-gradients/concat_1_grad/ConcatOffset:offset:0'gradients/concat_1_grad/ShapeN:output:0*
Index0*
T0*4
_output_shapes"
 :������������������ �
gradients/concat_1_grad/Slice_1Slice(gradients/MatMul_1_grad/Reshape:output:0-gradients/concat_1_grad/ConcatOffset:offset:1'gradients/concat_1_grad/ShapeN:output:1*
Index0*
T0*4
_output_shapes"
 :������������������ \
gradients/concat_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: n
gradients/concat_grad/ShapeShape#gradients_concat_grad_shape_dst_neg*
T0*
_output_shapes
:�
gradients/concat_grad/ShapeNShapeN#gradients_concat_grad_shape_dst_neg!gradients_concat_grad_shapen_mean*
N*
T0* 
_output_shapes
::�
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0%gradients/concat_grad/ShapeN:output:0%gradients/concat_grad/ShapeN:output:1*
N* 
_output_shapes
::�
gradients/concat_grad/SliceSlice&gradients/MatMul_grad/Reshape:output:0+gradients/concat_grad/ConcatOffset:offset:0%gradients/concat_grad/ShapeN:output:0*
Index0*
T0*4
_output_shapes"
 :������������������ �
gradients/concat_grad/Slice_1Slice&gradients/MatMul_grad/Reshape:output:0+gradients/concat_grad/ConcatOffset:offset:1%gradients/concat_grad/ShapeN:output:1*
Index0*
T0*4
_output_shapes"
 :������������������ r
gradients/Mean_1_grad/ShapeShape'gradients_mean_1_grad_shape_src_neg_neg*
T0*
_output_shapes
:�
gradients/Mean_1_grad/SizeConst*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_1_grad/addAddV22gradients_mean_1_grad_add_mean_1_reduction_indices#gradients/Mean_1_grad/Size:output:0*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: �
gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/add:z:0#gradients/Mean_1_grad/Size:output:0*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: �
gradients/Mean_1_grad/Shape_1Const*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
valueB �
!gradients/Mean_1_grad/range/startConst*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B : �
!gradients/Mean_1_grad/range/deltaConst*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_1_grad/rangeRange*gradients/Mean_1_grad/range/start:output:0#gradients/Mean_1_grad/Size:output:0*gradients/Mean_1_grad/range/delta:output:0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
:�
 gradients/Mean_1_grad/ones/ConstConst*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_1_grad/onesFill&gradients/Mean_1_grad/Shape_1:output:0)gradients/Mean_1_grad/ones/Const:output:0*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*
_output_shapes
: �
#gradients/Mean_1_grad/DynamicStitchDynamicStitch$gradients/Mean_1_grad/range:output:0gradients/Mean_1_grad/mod:z:0$gradients/Mean_1_grad/Shape:output:0#gradients/Mean_1_grad/ones:output:0*
N*
T0*.
_class$
" loc:@gradients/Mean_1_grad/Shape*#
_output_shapes
:����������
gradients/Mean_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0,gradients/Mean_1_grad/DynamicStitch:merged:0*
T0*
_output_shapes
:�
!gradients/Mean_1_grad/BroadcastToBroadcastTo&gradients/Mean_1_grad/Reshape:output:0$gradients/Mean_1_grad/Shape:output:0*
T0*A
_output_shapes/
-:+��������������������������� t
gradients/Mean_1_grad/Shape_2Shape'gradients_mean_1_grad_shape_src_neg_neg*
T0*
_output_shapes
:r
gradients/Mean_1_grad/Shape_3Shape%gradients_concat_1_grad_shapen_mean_1*
T0*
_output_shapes
:e
gradients/Mean_1_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_1_grad/ProdProd&gradients/Mean_1_grad/Shape_2:output:0$gradients/Mean_1_grad/Const:output:0*
T0*
_output_shapes
: g
gradients/Mean_1_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_1_grad/Prod_1Prod&gradients/Mean_1_grad/Shape_3:output:0&gradients/Mean_1_grad/Const_1:output:0*
T0*
_output_shapes
: a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_1_grad/MaximumMaximum%gradients/Mean_1_grad/Prod_1:output:0(gradients/Mean_1_grad/Maximum/y:output:0*
T0*
_output_shapes
: �
gradients/Mean_1_grad/floordivFloorDiv#gradients/Mean_1_grad/Prod:output:0!gradients/Mean_1_grad/Maximum:z:0*
T0*
_output_shapes
: v
gradients/Mean_1_grad/CastCast"gradients/Mean_1_grad/floordiv:z:0*

DstT0*

SrcT0*
_output_shapes
: �
gradients/Mean_1_grad/truedivRealDiv*gradients/Mean_1_grad/BroadcastTo:output:0gradients/Mean_1_grad/Cast:y:0*
T0*A
_output_shapes/
-:+��������������������������� n
gradients/Mean_grad/ShapeShape%gradients_mean_grad_shape_dst_neg_neg*
T0*
_output_shapes
:�
gradients/Mean_grad/SizeConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_grad/addAddV2.gradients_mean_grad_add_mean_reduction_indices!gradients/Mean_grad/Size:output:0*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: �
gradients/Mean_grad/modFloorModgradients/Mean_grad/add:z:0!gradients/Mean_grad/Size:output:0*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: �
gradients/Mean_grad/Shape_1Const*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
dtype0*
valueB �
gradients/Mean_grad/range/startConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B : �
gradients/Mean_grad/range/deltaConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_grad/rangeRange(gradients/Mean_grad/range/start:output:0!gradients/Mean_grad/Size:output:0(gradients/Mean_grad/range/delta:output:0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
:�
gradients/Mean_grad/ones/ConstConst*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_grad/onesFill$gradients/Mean_grad/Shape_1:output:0'gradients/Mean_grad/ones/Const:output:0*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*
_output_shapes
: �
!gradients/Mean_grad/DynamicStitchDynamicStitch"gradients/Mean_grad/range:output:0gradients/Mean_grad/mod:z:0"gradients/Mean_grad/Shape:output:0!gradients/Mean_grad/ones:output:0*
N*
T0*,
_class"
 loc:@gradients/Mean_grad/Shape*#
_output_shapes
:����������
gradients/Mean_grad/ReshapeReshape&gradients/concat_grad/Slice_1:output:0*gradients/Mean_grad/DynamicStitch:merged:0*
T0*
_output_shapes
:�
gradients/Mean_grad/BroadcastToBroadcastTo$gradients/Mean_grad/Reshape:output:0"gradients/Mean_grad/Shape:output:0*
T0*A
_output_shapes/
-:+��������������������������� p
gradients/Mean_grad/Shape_2Shape%gradients_mean_grad_shape_dst_neg_neg*
T0*
_output_shapes
:l
gradients/Mean_grad/Shape_3Shape!gradients_concat_grad_shapen_mean*
T0*
_output_shapes
:c
gradients/Mean_grad/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_grad/ProdProd$gradients/Mean_grad/Shape_2:output:0"gradients/Mean_grad/Const:output:0*
T0*
_output_shapes
: e
gradients/Mean_grad/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
gradients/Mean_grad/Prod_1Prod$gradients/Mean_grad/Shape_3:output:0$gradients/Mean_grad/Const_1:output:0*
T0*
_output_shapes
: _
gradients/Mean_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :�
gradients/Mean_grad/MaximumMaximum#gradients/Mean_grad/Prod_1:output:0&gradients/Mean_grad/Maximum/y:output:0*
T0*
_output_shapes
: �
gradients/Mean_grad/floordivFloorDiv!gradients/Mean_grad/Prod:output:0gradients/Mean_grad/Maximum:z:0*
T0*
_output_shapes
: r
gradients/Mean_grad/CastCast gradients/Mean_grad/floordiv:z:0*

DstT0*

SrcT0*
_output_shapes
: �
gradients/Mean_grad/truedivRealDiv(gradients/Mean_grad/BroadcastTo:output:0gradients/Mean_grad/Cast:y:0*
T0*A
_output_shapes/
-:+��������������������������� �
gradients/AddN_1AddN*gradients/MatMul_1_grad/Reshape_1:output:0(gradients/MatMul_grad/Reshape_1:output:0*
N*
T0*4
_class*
(&loc:@gradients/MatMul_1_grad/Reshape_1*
_output_shapes

:@ n
IdentityIdentity&gradients/concat_3_grad/Slice:output:0*
T0*'
_output_shapes
:��������� }

Identity_1Identity&gradients/concat_1_grad/Slice:output:0*
T0*4
_output_shapes"
 :������������������ �

Identity_2Identity!gradients/Mean_1_grad/truediv:z:0*
T0*A
_output_shapes/
-:+��������������������������� p

Identity_3Identity&gradients/concat_2_grad/Slice:output:0*
T0*'
_output_shapes
:��������� {

Identity_4Identity$gradients/concat_grad/Slice:output:0*
T0*4
_output_shapes"
 :������������������ �

Identity_5Identitygradients/Mean_grad/truediv:z:0*
T0*A
_output_shapes/
-:+��������������������������� W

Identity_6Identitygradients/AddN_1:sum:0*
T0*
_output_shapes

:@ U

Identity_7Identitygradients/AddN:sum:0*
T0*
_output_shapes

:@ o

Identity_8Identity.gradients/dense/MatMul_grad/MatMul_1:product:0*
T0*
_output_shapes

:@"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:@:���������@: :��������� :��������� :@ :���������@:@ :���������@: :��������� :��������� : :��������� :��������� :������������������ : :������������������ : :@ :������������������@:@ :������������������@: :������������������ :������������������ : :������������������ :������������������ :+��������������������������� : :+��������������������������� : */
forward_function_name__forward_call_16059:- )
'
_output_shapes
:���������:-)
'
_output_shapes
:���������:$ 

_output_shapes

:@:-)
'
_output_shapes
:���������@:

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :$ 

_output_shapes

:@ :-)
'
_output_shapes
:���������@:$	 

_output_shapes

:@ :-
)
'
_output_shapes
:���������@:

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� :

_output_shapes
: :-)
'
_output_shapes
:��������� :-)
'
_output_shapes
:��������� ::6
4
_output_shapes"
 :������������������ :LH
.
_class$
" loc:@gradients/Mean_3_grad/Shape

_output_shapes
: ::6
4
_output_shapes"
 :������������������ :LH
.
_class$
" loc:@gradients/Mean_2_grad/Shape

_output_shapes
: :$ 

_output_shapes

:@ ::6
4
_output_shapes"
 :������������������@:$ 

_output_shapes

:@ ::6
4
_output_shapes"
 :������������������@:

_output_shapes
: ::6
4
_output_shapes"
 :������������������ ::6
4
_output_shapes"
 :������������������ :

_output_shapes
: ::6
4
_output_shapes"
 :������������������ ::6
4
_output_shapes"
 :������������������ :GC
A
_output_shapes/
-:+��������������������������� :L H
.
_class$
" loc:@gradients/Mean_1_grad/Shape

_output_shapes
: :G!C
A
_output_shapes/
-:+��������������������������� :J"F
,
_class"
 loc:@gradients/Mean_grad/Shape

_output_shapes
: 
� 
�
!__inference__traced_restore_16228
file_prefix/
assignvariableop_dense_kernel:@%
assignvariableop_1_sgd_iter:	 &
assignvariableop_2_sgd_decay: .
$assignvariableop_3_sgd_learning_rate: )
assignvariableop_4_sgd_momentum: D
2assignvariableop_5_mean_aggregation_layer_1_weight:@ D
2assignvariableop_6_mean_aggregation_layer_2_weight:@ 

identity_8��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEBGseq_layers/0/mean_aggregation_layer_1_weight/.ATTRIBUTES/VARIABLE_VALUEBGseq_layers/1/mean_aggregation_layer_2_weight/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*#
valueBB B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*4
_output_shapes"
 ::::::::*
dtypes

2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_sgd_iterIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_sgd_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp$assignvariableop_3_sgd_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_momentumIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp2assignvariableop_5_mean_aggregation_layer_1_weightIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_mean_aggregation_layer_2_weightIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_7Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_8IdentityIdentity_7:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6*"
_acd_function_control_output(*
_output_shapes
 "!

identity_8Identity_8:output:0*#
_input_shapes
: : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_6:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�6
�
__forward_call_16059	
src_0
	src_neg_0
src_neg_neg_0	
dst_0
	dst_neg_0
dst_neg_neg_00
matmul_readvariableop_resource:@ 2
 matmul_2_readvariableop_resource:@ 6
$dense_matmul_readvariableop_resource:@
identity

dense_relu
dense_matmul_readvariableop
concat_4
concat_4_axis
matmul_3
matmul_2
matmul_3_readvariableop
concat_3
matmul_2_readvariableop
concat_2
concat_3_axis
src

mean_3
concat_2_axis
dst

mean_2
matmul_1
mean_3_reduction_indices

matmul
mean_2_reduction_indices
matmul_1_readvariableop
concat_1
matmul_readvariableop

concat
concat_1_axis
src_neg

mean_1
concat_axis
dst_neg
mean
src_neg_neg
mean_1_reduction_indices
dst_neg_neg
mean_reduction_indices��MatMul/ReadVariableOp�MatMul_1/ReadVariableOp�MatMul_2/ReadVariableOp�MatMul_3/ReadVariableOp�dense/MatMul/ReadVariableOpX
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :{
MeanMeandst_neg_neg_0Mean/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������ Z
Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
Mean_1Meansrc_neg_neg_0!Mean_1/reduction_indices:output:0*
T0*4
_output_shapes"
 :������������������ M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :V
concat_0ConcatV2	dst_neg_0Mean:output:0concat/axis:output:0*
N*
T0O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B :\

concat_1_0ConcatV2	src_neg_0Mean_1:output:0concat_1/axis:output:0*
N*
T0t
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
MatMulBatchMatMulV2concat_0:output:0MatMul/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ v
MatMul_1/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@ *
dtype0�
MatMul_1BatchMatMulV2concat_1_0:output:0MatMul_1/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������ Z
Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :t
Mean_2MeanMatMul:output:0!Mean_2/reduction_indices:output:0*
T0*'
_output_shapes
:��������� Z
Mean_3/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :v
Mean_3MeanMatMul_1:output:0!Mean_3/reduction_indices:output:0*
T0*'
_output_shapes
:��������� O
concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B :X

concat_2_0ConcatV2dst_0Mean_2:output:0concat_2/axis:output:0*
N*
T0O
concat_3/axisConst*
_output_shapes
: *
dtype0*
value	B :X

concat_3_0ConcatV2src_0Mean_3:output:0concat_3/axis:output:0*
N*
T0x
MatMul_2/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@ *
dtype0z
MatMul_2MatMulconcat_2_0:output:0MatMul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� x
MatMul_3/ReadVariableOpReadVariableOp matmul_2_readvariableop_resource*
_output_shapes

:@ *
dtype0z
MatMul_3MatMulconcat_3_0:output:0MatMul_3/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� O
concat_4/axisConst*
_output_shapes
: *
dtype0*
value	B :h

concat_4_0ConcatV2MatMul_3:product:0MatMul_2:product:0concat_4/axis:output:0*
N*
T0�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
dense/MatMulMatMulconcat_4_0:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������\

dense/ReluReludense/MatMul:product:0*
T0*'
_output_shapes
:���������g
IdentityIdentitydense/Relu:activations:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^MatMul_2/ReadVariableOp^MatMul_3/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
concatconcat_0:output:0"
concat_1concat_1_0:output:0"'
concat_1_axisconcat_1/axis:output:0"
concat_2concat_2_0:output:0"'
concat_2_axisconcat_2/axis:output:0"
concat_3concat_3_0:output:0"'
concat_3_axisconcat_3/axis:output:0"
concat_4concat_4_0:output:0"'
concat_4_axisconcat_4/axis:output:0"#
concat_axisconcat/axis:output:0"B
dense_matmul_readvariableop#dense/MatMul/ReadVariableOp:value:0"&

dense_reludense/Relu:activations:0"
dstdst_0"
dst_neg	dst_neg_0"
dst_neg_negdst_neg_neg_0"
identityIdentity:output:0"
matmulMatMul:output:0"
matmul_1MatMul_1:output:0":
matmul_1_readvariableopMatMul_1/ReadVariableOp:value:0"
matmul_2MatMul_2:product:0":
matmul_2_readvariableopMatMul_2/ReadVariableOp:value:0"
matmul_3MatMul_3:product:0":
matmul_3_readvariableopMatMul_3/ReadVariableOp:value:0"6
matmul_readvariableopMatMul/ReadVariableOp:value:0"
meanMean:output:0"
mean_1Mean_1:output:0"=
mean_1_reduction_indices!Mean_1/reduction_indices:output:0"
mean_2Mean_2:output:0"=
mean_2_reduction_indices!Mean_2/reduction_indices:output:0"
mean_3Mean_3:output:0"=
mean_3_reduction_indices!Mean_3/reduction_indices:output:0"9
mean_reduction_indicesMean/reduction_indices:output:0"
srcsrc_0"
src_neg	src_neg_0"
src_neg_negsrc_neg_neg_0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:��������� :������������������ :+��������������������������� :��������� :������������������ :+��������������������������� : : : *C
backward_function_name)'__inference___backward_call_15826_160602.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp22
MatMul_2/ReadVariableOpMatMul_2/ReadVariableOp22
MatMul_3/ReadVariableOpMatMul_3/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:L H
'
_output_shapes
:��������� 

_user_specified_namesrc:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	src_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namesrc_neg_neg:LH
'
_output_shapes
:��������� 

_user_specified_namedst:]Y
4
_output_shapes"
 :������������������ 
!
_user_specified_name	dst_neg:nj
A
_output_shapes/
-:+��������������������������� 
%
_user_specified_namedst_neg_neg:
"__inference_internal_grad_fn_16171CustomGradient-16110"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
call�
(
dst!

call_dst:0��������� 
=
dst_neg2
call_dst_neg:0������������������ 
R
dst_neg_negC
call_dst_neg_neg:0+��������������������������� 
(
src!

call_src:0��������� 
=
src_neg2
call_src_neg:0������������������ 
R
src_neg_negC
call_src_neg_neg:0+��������������������������� <
output_00
StatefulPartitionedCall:0���������tensorflow/serving/predict:�%
�
samples

seq_layers
	dense
	optimizer
	keras_api

signatures
call
	train"
_tf_keras_model
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�

	kernel

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
I
iter
	decay
learning_rate
momentum"
	optimizer
"
_generic_user_object
!
call"
signature_map
T
#mean_aggregation_layer_1_weight
w
	keras_api"
_tf_keras_layer
T
#mean_aggregation_layer_2_weight
w
	keras_api"
_tf_keras_layer
:@2dense/kernel
'
	0"
trackable_list_wrapper
'
	0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics

	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
1:/@ 2mean_aggregation_layer_1_weight
"
_generic_user_object
1:/@ 2mean_aggregation_layer_2_weight
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
__inference_call_15718�
���
FullArgSpecU
argsM�J
jself
jsrc
	jsrc_neg
jsrc_neg_neg
jdst
	jdst_neg
jdst_neg_neg
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���������� 
%�"������������������ 
2�/+��������������������������� 
���������� 
%�"������������������ 
2�/+��������������������������� 
�2�
__inference_train_16126�
���
FullArgSpec_
argsW�T
jself
jsrc
	jsrc_neg
jsrc_neg_neg
jdst
	jdst_neg
jdst_neg_neg
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *���
���������� 
%�"������������������ 
2�/+��������������������������� 
���������� 
%�"������������������ 
2�/+��������������������������� 
����������
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_15736dstdst_negdst_neg_negsrcsrc_negsrc_neg_neg"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference_call_15718�	���
���
�
src��������� 
.�+
src_neg������������������ 
?�<
src_neg_neg+��������������������������� 
�
dst��������� 
.�+
dst_neg������������������ 
?�<
dst_neg_neg+��������������������������� 
� "�����������
"__inference_internal_grad_fn_16171����
���

 
�
result_grads_0@ 
�
result_grads_1@ 
�
result_grads_2@
�
result_grads_3@ 
�
result_grads_4@ 
�
result_grads_5@
� "K�H

 

 

 
�
3@ 
�
4@ 
�
5@�
#__inference_signature_wrapper_15736�	���
� 
���
$
dst�
dst��������� 
9
dst_neg.�+
dst_neg������������������ 
N
dst_neg_neg?�<
dst_neg_neg+��������������������������� 
$
src�
src��������� 
9
src_neg.�+
src_neg������������������ 
N
src_neg_neg?�<
src_neg_neg+��������������������������� "3�0
.
output_0"�
output_0����������
__inference_train_16126�	���
���
�
src��������� 
.�+
src_neg������������������ 
?�<
src_neg_neg+��������������������������� 
�
dst��������� 
.�+
dst_neg������������������ 
?�<
dst_neg_neg+��������������������������� 
 �
labels���������
� "� 