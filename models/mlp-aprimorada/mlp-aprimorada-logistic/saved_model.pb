��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
executor_typestring �
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
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
�
camada_de_entrada/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_namecamada_de_entrada/kernel
�
,camada_de_entrada/kernel/Read/ReadVariableOpReadVariableOpcamada_de_entrada/kernel*
_output_shapes

:*
dtype0
�
camada_de_entrada/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namecamada_de_entrada/bias
}
*camada_de_entrada/bias/Read/ReadVariableOpReadVariableOpcamada_de_entrada/bias*
_output_shapes
:*
dtype0
�
camada_intermediaria_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*.
shared_namecamada_intermediaria_1/kernel
�
1camada_intermediaria_1/kernel/Read/ReadVariableOpReadVariableOpcamada_intermediaria_1/kernel*
_output_shapes

:
*
dtype0
�
camada_intermediaria_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namecamada_intermediaria_1/bias
�
/camada_intermediaria_1/bias/Read/ReadVariableOpReadVariableOpcamada_intermediaria_1/bias*
_output_shapes
:
*
dtype0
�
camada_intermediaria_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*.
shared_namecamada_intermediaria_2/kernel
�
1camada_intermediaria_2/kernel/Read/ReadVariableOpReadVariableOpcamada_intermediaria_2/kernel*
_output_shapes

:

*
dtype0
�
camada_intermediaria_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namecamada_intermediaria_2/bias
�
/camada_intermediaria_2/bias/Read/ReadVariableOpReadVariableOpcamada_intermediaria_2/bias*
_output_shapes
:
*
dtype0
�
camada_intermediaria_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*.
shared_namecamada_intermediaria_3/kernel
�
1camada_intermediaria_3/kernel/Read/ReadVariableOpReadVariableOpcamada_intermediaria_3/kernel*
_output_shapes

:

*
dtype0
�
camada_intermediaria_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*,
shared_namecamada_intermediaria_3/bias
�
/camada_intermediaria_3/bias/Read/ReadVariableOpReadVariableOpcamada_intermediaria_3/bias*
_output_shapes
:
*
dtype0
�
camada_de_saida/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*'
shared_namecamada_de_saida/kernel
�
*camada_de_saida/kernel/Read/ReadVariableOpReadVariableOpcamada_de_saida/kernel*
_output_shapes

:
*
dtype0
�
camada_de_saida/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namecamada_de_saida/bias
y
(camada_de_saida/bias/Read/ReadVariableOpReadVariableOpcamada_de_saida/bias*
_output_shapes
:*
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
h

Nadam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
Nadam/iter
a
Nadam/iter/Read/ReadVariableOpReadVariableOp
Nadam/iter*
_output_shapes
: *
dtype0	
|
Nadam/momentum_cacheVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameNadam/momentum_cache
u
(Nadam/momentum_cache/Read/ReadVariableOpReadVariableOpNadam/momentum_cache*
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
�
 Nadam/camada_de_entrada/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Nadam/camada_de_entrada/kernel/m
�
4Nadam/camada_de_entrada/kernel/m/Read/ReadVariableOpReadVariableOp Nadam/camada_de_entrada/kernel/m*
_output_shapes

:*
dtype0
�
Nadam/camada_de_entrada/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/camada_de_entrada/bias/m
�
2Nadam/camada_de_entrada/bias/m/Read/ReadVariableOpReadVariableOpNadam/camada_de_entrada/bias/m*
_output_shapes
:*
dtype0
�
%Nadam/camada_intermediaria_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%Nadam/camada_intermediaria_1/kernel/m
�
9Nadam/camada_intermediaria_1/kernel/m/Read/ReadVariableOpReadVariableOp%Nadam/camada_intermediaria_1/kernel/m*
_output_shapes

:
*
dtype0
�
#Nadam/camada_intermediaria_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Nadam/camada_intermediaria_1/bias/m
�
7Nadam/camada_intermediaria_1/bias/m/Read/ReadVariableOpReadVariableOp#Nadam/camada_intermediaria_1/bias/m*
_output_shapes
:
*
dtype0
�
%Nadam/camada_intermediaria_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*6
shared_name'%Nadam/camada_intermediaria_2/kernel/m
�
9Nadam/camada_intermediaria_2/kernel/m/Read/ReadVariableOpReadVariableOp%Nadam/camada_intermediaria_2/kernel/m*
_output_shapes

:

*
dtype0
�
#Nadam/camada_intermediaria_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Nadam/camada_intermediaria_2/bias/m
�
7Nadam/camada_intermediaria_2/bias/m/Read/ReadVariableOpReadVariableOp#Nadam/camada_intermediaria_2/bias/m*
_output_shapes
:
*
dtype0
�
%Nadam/camada_intermediaria_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*6
shared_name'%Nadam/camada_intermediaria_3/kernel/m
�
9Nadam/camada_intermediaria_3/kernel/m/Read/ReadVariableOpReadVariableOp%Nadam/camada_intermediaria_3/kernel/m*
_output_shapes

:

*
dtype0
�
#Nadam/camada_intermediaria_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Nadam/camada_intermediaria_3/bias/m
�
7Nadam/camada_intermediaria_3/bias/m/Read/ReadVariableOpReadVariableOp#Nadam/camada_intermediaria_3/bias/m*
_output_shapes
:
*
dtype0
�
Nadam/camada_de_saida/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*/
shared_name Nadam/camada_de_saida/kernel/m
�
2Nadam/camada_de_saida/kernel/m/Read/ReadVariableOpReadVariableOpNadam/camada_de_saida/kernel/m*
_output_shapes

:
*
dtype0
�
Nadam/camada_de_saida/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameNadam/camada_de_saida/bias/m
�
0Nadam/camada_de_saida/bias/m/Read/ReadVariableOpReadVariableOpNadam/camada_de_saida/bias/m*
_output_shapes
:*
dtype0
�
 Nadam/camada_de_entrada/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" Nadam/camada_de_entrada/kernel/v
�
4Nadam/camada_de_entrada/kernel/v/Read/ReadVariableOpReadVariableOp Nadam/camada_de_entrada/kernel/v*
_output_shapes

:*
dtype0
�
Nadam/camada_de_entrada/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Nadam/camada_de_entrada/bias/v
�
2Nadam/camada_de_entrada/bias/v/Read/ReadVariableOpReadVariableOpNadam/camada_de_entrada/bias/v*
_output_shapes
:*
dtype0
�
%Nadam/camada_intermediaria_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*6
shared_name'%Nadam/camada_intermediaria_1/kernel/v
�
9Nadam/camada_intermediaria_1/kernel/v/Read/ReadVariableOpReadVariableOp%Nadam/camada_intermediaria_1/kernel/v*
_output_shapes

:
*
dtype0
�
#Nadam/camada_intermediaria_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Nadam/camada_intermediaria_1/bias/v
�
7Nadam/camada_intermediaria_1/bias/v/Read/ReadVariableOpReadVariableOp#Nadam/camada_intermediaria_1/bias/v*
_output_shapes
:
*
dtype0
�
%Nadam/camada_intermediaria_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*6
shared_name'%Nadam/camada_intermediaria_2/kernel/v
�
9Nadam/camada_intermediaria_2/kernel/v/Read/ReadVariableOpReadVariableOp%Nadam/camada_intermediaria_2/kernel/v*
_output_shapes

:

*
dtype0
�
#Nadam/camada_intermediaria_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Nadam/camada_intermediaria_2/bias/v
�
7Nadam/camada_intermediaria_2/bias/v/Read/ReadVariableOpReadVariableOp#Nadam/camada_intermediaria_2/bias/v*
_output_shapes
:
*
dtype0
�
%Nadam/camada_intermediaria_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:

*6
shared_name'%Nadam/camada_intermediaria_3/kernel/v
�
9Nadam/camada_intermediaria_3/kernel/v/Read/ReadVariableOpReadVariableOp%Nadam/camada_intermediaria_3/kernel/v*
_output_shapes

:

*
dtype0
�
#Nadam/camada_intermediaria_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*4
shared_name%#Nadam/camada_intermediaria_3/bias/v
�
7Nadam/camada_intermediaria_3/bias/v/Read/ReadVariableOpReadVariableOp#Nadam/camada_intermediaria_3/bias/v*
_output_shapes
:
*
dtype0
�
Nadam/camada_de_saida/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:
*/
shared_name Nadam/camada_de_saida/kernel/v
�
2Nadam/camada_de_saida/kernel/v/Read/ReadVariableOpReadVariableOpNadam/camada_de_saida/kernel/v*
_output_shapes

:
*
dtype0
�
Nadam/camada_de_saida/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameNadam/camada_de_saida/bias/v
�
0Nadam/camada_de_saida/bias/v/Read/ReadVariableOpReadVariableOpNadam/camada_de_saida/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�7
value�7B�7 B�7
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
�

*beta_1

+beta_2
	,decay
-learning_rate
.iter
/momentum_cachemSmTmUmVmWmXmYmZ$m[%m\v]v^v_v`vavbvcvd$ve%vf
 
F
0
1
2
3
4
5
6
7
$8
%9
F
0
1
2
3
4
5
6
7
$8
%9
�
regularization_losses
0layer_metrics

1layers
2non_trainable_variables
3metrics
4layer_regularization_losses
	variables
	trainable_variables
 
db
VARIABLE_VALUEcamada_de_entrada/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEcamada_de_entrada/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
5layer_metrics

6layers
7non_trainable_variables
8metrics
9layer_regularization_losses
	variables
trainable_variables
ig
VARIABLE_VALUEcamada_intermediaria_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEcamada_intermediaria_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
:layer_metrics

;layers
<non_trainable_variables
=metrics
>layer_regularization_losses
	variables
trainable_variables
ig
VARIABLE_VALUEcamada_intermediaria_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEcamada_intermediaria_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses
?layer_metrics

@layers
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
	variables
trainable_variables
ig
VARIABLE_VALUEcamada_intermediaria_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEcamada_intermediaria_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
 regularization_losses
Dlayer_metrics

Elayers
Fnon_trainable_variables
Gmetrics
Hlayer_regularization_losses
!	variables
"trainable_variables
b`
VARIABLE_VALUEcamada_de_saida/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEcamada_de_saida/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
�
&regularization_losses
Ilayer_metrics

Jlayers
Knon_trainable_variables
Lmetrics
Mlayer_regularization_losses
'	variables
(trainable_variables
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE
Nadam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUENadam/momentum_cache3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

N0
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
	Ototal
	Pcount
Q	variables
R	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

O0
P1

Q	variables
��
VARIABLE_VALUE Nadam/camada_de_entrada/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/camada_de_entrada/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Nadam/camada_intermediaria_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Nadam/camada_intermediaria_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Nadam/camada_intermediaria_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Nadam/camada_intermediaria_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Nadam/camada_intermediaria_3/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Nadam/camada_intermediaria_3/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/camada_de_saida/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/camada_de_saida/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Nadam/camada_de_entrada/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/camada_de_entrada/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Nadam/camada_intermediaria_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Nadam/camada_intermediaria_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Nadam/camada_intermediaria_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Nadam/camada_intermediaria_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE%Nadam/camada_intermediaria_3/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE#Nadam/camada_intermediaria_3/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/camada_de_saida/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUENadam/camada_de_saida/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
'serving_default_camada_de_entrada_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCall'serving_default_camada_de_entrada_inputcamada_de_entrada/kernelcamada_de_entrada/biascamada_intermediaria_1/kernelcamada_intermediaria_1/biascamada_intermediaria_2/kernelcamada_intermediaria_2/biascamada_intermediaria_3/kernelcamada_intermediaria_3/biascamada_de_saida/kernelcamada_de_saida/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_367374
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,camada_de_entrada/kernel/Read/ReadVariableOp*camada_de_entrada/bias/Read/ReadVariableOp1camada_intermediaria_1/kernel/Read/ReadVariableOp/camada_intermediaria_1/bias/Read/ReadVariableOp1camada_intermediaria_2/kernel/Read/ReadVariableOp/camada_intermediaria_2/bias/Read/ReadVariableOp1camada_intermediaria_3/kernel/Read/ReadVariableOp/camada_intermediaria_3/bias/Read/ReadVariableOp*camada_de_saida/kernel/Read/ReadVariableOp(camada_de_saida/bias/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpNadam/iter/Read/ReadVariableOp(Nadam/momentum_cache/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp4Nadam/camada_de_entrada/kernel/m/Read/ReadVariableOp2Nadam/camada_de_entrada/bias/m/Read/ReadVariableOp9Nadam/camada_intermediaria_1/kernel/m/Read/ReadVariableOp7Nadam/camada_intermediaria_1/bias/m/Read/ReadVariableOp9Nadam/camada_intermediaria_2/kernel/m/Read/ReadVariableOp7Nadam/camada_intermediaria_2/bias/m/Read/ReadVariableOp9Nadam/camada_intermediaria_3/kernel/m/Read/ReadVariableOp7Nadam/camada_intermediaria_3/bias/m/Read/ReadVariableOp2Nadam/camada_de_saida/kernel/m/Read/ReadVariableOp0Nadam/camada_de_saida/bias/m/Read/ReadVariableOp4Nadam/camada_de_entrada/kernel/v/Read/ReadVariableOp2Nadam/camada_de_entrada/bias/v/Read/ReadVariableOp9Nadam/camada_intermediaria_1/kernel/v/Read/ReadVariableOp7Nadam/camada_intermediaria_1/bias/v/Read/ReadVariableOp9Nadam/camada_intermediaria_2/kernel/v/Read/ReadVariableOp7Nadam/camada_intermediaria_2/bias/v/Read/ReadVariableOp9Nadam/camada_intermediaria_3/kernel/v/Read/ReadVariableOp7Nadam/camada_intermediaria_3/bias/v/Read/ReadVariableOp2Nadam/camada_de_saida/kernel/v/Read/ReadVariableOp0Nadam/camada_de_saida/bias/v/Read/ReadVariableOpConst*3
Tin,
*2(	*
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
GPU2*0J 8� *(
f#R!
__inference__traced_save_367733
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamecamada_de_entrada/kernelcamada_de_entrada/biascamada_intermediaria_1/kernelcamada_intermediaria_1/biascamada_intermediaria_2/kernelcamada_intermediaria_2/biascamada_intermediaria_3/kernelcamada_intermediaria_3/biascamada_de_saida/kernelcamada_de_saida/biasbeta_1beta_2decaylearning_rate
Nadam/iterNadam/momentum_cachetotalcount Nadam/camada_de_entrada/kernel/mNadam/camada_de_entrada/bias/m%Nadam/camada_intermediaria_1/kernel/m#Nadam/camada_intermediaria_1/bias/m%Nadam/camada_intermediaria_2/kernel/m#Nadam/camada_intermediaria_2/bias/m%Nadam/camada_intermediaria_3/kernel/m#Nadam/camada_intermediaria_3/bias/mNadam/camada_de_saida/kernel/mNadam/camada_de_saida/bias/m Nadam/camada_de_entrada/kernel/vNadam/camada_de_entrada/bias/v%Nadam/camada_intermediaria_1/kernel/v#Nadam/camada_intermediaria_1/bias/v%Nadam/camada_intermediaria_2/kernel/v#Nadam/camada_intermediaria_2/bias/v%Nadam/camada_intermediaria_3/kernel/v#Nadam/camada_intermediaria_3/bias/vNadam/camada_de_saida/kernel/vNadam/camada_de_saida/bias/v*2
Tin+
)2'*
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
GPU2*0J 8� *+
f&R$
"__inference__traced_restore_367857��
�
�
-__inference_MLP-logistic_layer_call_fn_367473

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_3672622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_367158

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_367528

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367230
camada_de_entrada_input
camada_de_entrada_367204
camada_de_entrada_367206!
camada_intermediaria_1_367209!
camada_intermediaria_1_367211!
camada_intermediaria_2_367214!
camada_intermediaria_2_367216!
camada_intermediaria_3_367219!
camada_intermediaria_3_367221
camada_de_saida_367224
camada_de_saida_367226
identity��)camada_de_entrada/StatefulPartitionedCall�'camada_de_saida/StatefulPartitionedCall�.camada_intermediaria_1/StatefulPartitionedCall�.camada_intermediaria_2/StatefulPartitionedCall�.camada_intermediaria_3/StatefulPartitionedCall�
)camada_de_entrada/StatefulPartitionedCallStatefulPartitionedCallcamada_de_entrada_inputcamada_de_entrada_367204camada_de_entrada_367206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_3670772+
)camada_de_entrada/StatefulPartitionedCall�
.camada_intermediaria_1/StatefulPartitionedCallStatefulPartitionedCall2camada_de_entrada/StatefulPartitionedCall:output:0camada_intermediaria_1_367209camada_intermediaria_1_367211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_36710420
.camada_intermediaria_1/StatefulPartitionedCall�
.camada_intermediaria_2/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_1/StatefulPartitionedCall:output:0camada_intermediaria_2_367214camada_intermediaria_2_367216*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_36713120
.camada_intermediaria_2/StatefulPartitionedCall�
.camada_intermediaria_3/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_2/StatefulPartitionedCall:output:0camada_intermediaria_3_367219camada_intermediaria_3_367221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_36715820
.camada_intermediaria_3/StatefulPartitionedCall�
'camada_de_saida/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_3/StatefulPartitionedCall:output:0camada_de_saida_367224camada_de_saida_367226*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_3671842)
'camada_de_saida/StatefulPartitionedCall�
IdentityIdentity0camada_de_saida/StatefulPartitionedCall:output:0*^camada_de_entrada/StatefulPartitionedCall(^camada_de_saida/StatefulPartitionedCall/^camada_intermediaria_1/StatefulPartitionedCall/^camada_intermediaria_2/StatefulPartitionedCall/^camada_intermediaria_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2V
)camada_de_entrada/StatefulPartitionedCall)camada_de_entrada/StatefulPartitionedCall2R
'camada_de_saida/StatefulPartitionedCall'camada_de_saida/StatefulPartitionedCall2`
.camada_intermediaria_1/StatefulPartitionedCall.camada_intermediaria_1/StatefulPartitionedCall2`
.camada_intermediaria_2/StatefulPartitionedCall.camada_intermediaria_2/StatefulPartitionedCall2`
.camada_intermediaria_3/StatefulPartitionedCall.camada_intermediaria_3/StatefulPartitionedCall:` \
'
_output_shapes
:���������
1
_user_specified_namecamada_de_entrada_input
�	
�
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_367568

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
��
�
"__inference__traced_restore_367857
file_prefix-
)assignvariableop_camada_de_entrada_kernel-
)assignvariableop_1_camada_de_entrada_bias4
0assignvariableop_2_camada_intermediaria_1_kernel2
.assignvariableop_3_camada_intermediaria_1_bias4
0assignvariableop_4_camada_intermediaria_2_kernel2
.assignvariableop_5_camada_intermediaria_2_bias4
0assignvariableop_6_camada_intermediaria_3_kernel2
.assignvariableop_7_camada_intermediaria_3_bias-
)assignvariableop_8_camada_de_saida_kernel+
'assignvariableop_9_camada_de_saida_bias
assignvariableop_10_beta_1
assignvariableop_11_beta_2
assignvariableop_12_decay%
!assignvariableop_13_learning_rate"
assignvariableop_14_nadam_iter,
(assignvariableop_15_nadam_momentum_cache
assignvariableop_16_total
assignvariableop_17_count8
4assignvariableop_18_nadam_camada_de_entrada_kernel_m6
2assignvariableop_19_nadam_camada_de_entrada_bias_m=
9assignvariableop_20_nadam_camada_intermediaria_1_kernel_m;
7assignvariableop_21_nadam_camada_intermediaria_1_bias_m=
9assignvariableop_22_nadam_camada_intermediaria_2_kernel_m;
7assignvariableop_23_nadam_camada_intermediaria_2_bias_m=
9assignvariableop_24_nadam_camada_intermediaria_3_kernel_m;
7assignvariableop_25_nadam_camada_intermediaria_3_bias_m6
2assignvariableop_26_nadam_camada_de_saida_kernel_m4
0assignvariableop_27_nadam_camada_de_saida_bias_m8
4assignvariableop_28_nadam_camada_de_entrada_kernel_v6
2assignvariableop_29_nadam_camada_de_entrada_bias_v=
9assignvariableop_30_nadam_camada_intermediaria_1_kernel_v;
7assignvariableop_31_nadam_camada_intermediaria_1_bias_v=
9assignvariableop_32_nadam_camada_intermediaria_2_kernel_v;
7assignvariableop_33_nadam_camada_intermediaria_2_bias_v=
9assignvariableop_34_nadam_camada_intermediaria_3_kernel_v;
7assignvariableop_35_nadam_camada_intermediaria_3_bias_v6
2assignvariableop_36_nadam_camada_de_saida_kernel_v4
0assignvariableop_37_nadam_camada_de_saida_bias_v
identity_39��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*�
value�B�'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp)assignvariableop_camada_de_entrada_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_camada_de_entrada_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp0assignvariableop_2_camada_intermediaria_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_camada_intermediaria_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp0assignvariableop_4_camada_intermediaria_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp.assignvariableop_5_camada_intermediaria_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp0assignvariableop_6_camada_intermediaria_3_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp.assignvariableop_7_camada_intermediaria_3_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp)assignvariableop_8_camada_de_saida_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_camada_de_saida_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_beta_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_beta_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_decayIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp!assignvariableop_13_learning_rateIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpassignvariableop_14_nadam_iterIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_nadam_momentum_cacheIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_nadam_camada_de_entrada_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp2assignvariableop_19_nadam_camada_de_entrada_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp9assignvariableop_20_nadam_camada_intermediaria_1_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp7assignvariableop_21_nadam_camada_intermediaria_1_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp9assignvariableop_22_nadam_camada_intermediaria_2_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp7assignvariableop_23_nadam_camada_intermediaria_2_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp9assignvariableop_24_nadam_camada_intermediaria_3_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp7assignvariableop_25_nadam_camada_intermediaria_3_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp2assignvariableop_26_nadam_camada_de_saida_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp0assignvariableop_27_nadam_camada_de_saida_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp4assignvariableop_28_nadam_camada_de_entrada_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp2assignvariableop_29_nadam_camada_de_entrada_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp9assignvariableop_30_nadam_camada_intermediaria_1_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp7assignvariableop_31_nadam_camada_intermediaria_1_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp9assignvariableop_32_nadam_camada_intermediaria_2_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp7assignvariableop_33_nadam_camada_intermediaria_2_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp9assignvariableop_34_nadam_camada_intermediaria_3_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp7assignvariableop_35_nadam_camada_intermediaria_3_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp2assignvariableop_36_nadam_camada_de_saida_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp0assignvariableop_37_nadam_camada_de_saida_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_379
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_38Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_38�
Identity_39IdentityIdentity_38:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_39"#
identity_39Identity_39:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372(
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
�	
�
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_367131

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
2__inference_camada_de_entrada_layer_call_fn_367517

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_3670772
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_367508

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_camada_intermediaria_2_layer_call_fn_367557

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_3671312
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_367184

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�
�
0__inference_camada_de_saida_layer_call_fn_367596

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_3671842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
� 
�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367316

inputs
camada_de_entrada_367290
camada_de_entrada_367292!
camada_intermediaria_1_367295!
camada_intermediaria_1_367297!
camada_intermediaria_2_367300!
camada_intermediaria_2_367302!
camada_intermediaria_3_367305!
camada_intermediaria_3_367307
camada_de_saida_367310
camada_de_saida_367312
identity��)camada_de_entrada/StatefulPartitionedCall�'camada_de_saida/StatefulPartitionedCall�.camada_intermediaria_1/StatefulPartitionedCall�.camada_intermediaria_2/StatefulPartitionedCall�.camada_intermediaria_3/StatefulPartitionedCall�
)camada_de_entrada/StatefulPartitionedCallStatefulPartitionedCallinputscamada_de_entrada_367290camada_de_entrada_367292*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_3670772+
)camada_de_entrada/StatefulPartitionedCall�
.camada_intermediaria_1/StatefulPartitionedCallStatefulPartitionedCall2camada_de_entrada/StatefulPartitionedCall:output:0camada_intermediaria_1_367295camada_intermediaria_1_367297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_36710420
.camada_intermediaria_1/StatefulPartitionedCall�
.camada_intermediaria_2/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_1/StatefulPartitionedCall:output:0camada_intermediaria_2_367300camada_intermediaria_2_367302*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_36713120
.camada_intermediaria_2/StatefulPartitionedCall�
.camada_intermediaria_3/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_2/StatefulPartitionedCall:output:0camada_intermediaria_3_367305camada_intermediaria_3_367307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_36715820
.camada_intermediaria_3/StatefulPartitionedCall�
'camada_de_saida/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_3/StatefulPartitionedCall:output:0camada_de_saida_367310camada_de_saida_367312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_3671842)
'camada_de_saida/StatefulPartitionedCall�
IdentityIdentity0camada_de_saida/StatefulPartitionedCall:output:0*^camada_de_entrada/StatefulPartitionedCall(^camada_de_saida/StatefulPartitionedCall/^camada_intermediaria_1/StatefulPartitionedCall/^camada_intermediaria_2/StatefulPartitionedCall/^camada_intermediaria_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2V
)camada_de_entrada/StatefulPartitionedCall)camada_de_entrada/StatefulPartitionedCall2R
'camada_de_saida/StatefulPartitionedCall'camada_de_saida/StatefulPartitionedCall2`
.camada_intermediaria_1/StatefulPartitionedCall.camada_intermediaria_1/StatefulPartitionedCall2`
.camada_intermediaria_2/StatefulPartitionedCall.camada_intermediaria_2/StatefulPartitionedCall2`
.camada_intermediaria_3/StatefulPartitionedCall.camada_intermediaria_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�G
�

!__inference__wrapped_model_367063
camada_de_entrada_inputA
=mlp_logistic_camada_de_entrada_matmul_readvariableop_resourceB
>mlp_logistic_camada_de_entrada_biasadd_readvariableop_resourceF
Bmlp_logistic_camada_intermediaria_1_matmul_readvariableop_resourceG
Cmlp_logistic_camada_intermediaria_1_biasadd_readvariableop_resourceF
Bmlp_logistic_camada_intermediaria_2_matmul_readvariableop_resourceG
Cmlp_logistic_camada_intermediaria_2_biasadd_readvariableop_resourceF
Bmlp_logistic_camada_intermediaria_3_matmul_readvariableop_resourceG
Cmlp_logistic_camada_intermediaria_3_biasadd_readvariableop_resource?
;mlp_logistic_camada_de_saida_matmul_readvariableop_resource@
<mlp_logistic_camada_de_saida_biasadd_readvariableop_resource
identity��5MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOp�4MLP-logistic/camada_de_entrada/MatMul/ReadVariableOp�3MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOp�2MLP-logistic/camada_de_saida/MatMul/ReadVariableOp�:MLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOp�9MLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOp�:MLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOp�9MLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOp�:MLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOp�9MLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOp�
4MLP-logistic/camada_de_entrada/MatMul/ReadVariableOpReadVariableOp=mlp_logistic_camada_de_entrada_matmul_readvariableop_resource*
_output_shapes

:*
dtype026
4MLP-logistic/camada_de_entrada/MatMul/ReadVariableOp�
%MLP-logistic/camada_de_entrada/MatMulMatMulcamada_de_entrada_input<MLP-logistic/camada_de_entrada/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2'
%MLP-logistic/camada_de_entrada/MatMul�
5MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOpReadVariableOp>mlp_logistic_camada_de_entrada_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOp�
&MLP-logistic/camada_de_entrada/BiasAddBiasAdd/MLP-logistic/camada_de_entrada/MatMul:product:0=MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&MLP-logistic/camada_de_entrada/BiasAdd�
9MLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOpReadVariableOpBmlp_logistic_camada_intermediaria_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02;
9MLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOp�
*MLP-logistic/camada_intermediaria_1/MatMulMatMul/MLP-logistic/camada_de_entrada/BiasAdd:output:0AMLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2,
*MLP-logistic/camada_intermediaria_1/MatMul�
:MLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOpReadVariableOpCmlp_logistic_camada_intermediaria_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02<
:MLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOp�
+MLP-logistic/camada_intermediaria_1/BiasAddBiasAdd4MLP-logistic/camada_intermediaria_1/MatMul:product:0BMLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2-
+MLP-logistic/camada_intermediaria_1/BiasAdd�
(MLP-logistic/camada_intermediaria_1/TanhTanh4MLP-logistic/camada_intermediaria_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2*
(MLP-logistic/camada_intermediaria_1/Tanh�
9MLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOpReadVariableOpBmlp_logistic_camada_intermediaria_2_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02;
9MLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOp�
*MLP-logistic/camada_intermediaria_2/MatMulMatMul,MLP-logistic/camada_intermediaria_1/Tanh:y:0AMLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2,
*MLP-logistic/camada_intermediaria_2/MatMul�
:MLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOpReadVariableOpCmlp_logistic_camada_intermediaria_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02<
:MLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOp�
+MLP-logistic/camada_intermediaria_2/BiasAddBiasAdd4MLP-logistic/camada_intermediaria_2/MatMul:product:0BMLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2-
+MLP-logistic/camada_intermediaria_2/BiasAdd�
(MLP-logistic/camada_intermediaria_2/TanhTanh4MLP-logistic/camada_intermediaria_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2*
(MLP-logistic/camada_intermediaria_2/Tanh�
9MLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOpReadVariableOpBmlp_logistic_camada_intermediaria_3_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02;
9MLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOp�
*MLP-logistic/camada_intermediaria_3/MatMulMatMul,MLP-logistic/camada_intermediaria_2/Tanh:y:0AMLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2,
*MLP-logistic/camada_intermediaria_3/MatMul�
:MLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOpReadVariableOpCmlp_logistic_camada_intermediaria_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02<
:MLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOp�
+MLP-logistic/camada_intermediaria_3/BiasAddBiasAdd4MLP-logistic/camada_intermediaria_3/MatMul:product:0BMLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2-
+MLP-logistic/camada_intermediaria_3/BiasAdd�
(MLP-logistic/camada_intermediaria_3/TanhTanh4MLP-logistic/camada_intermediaria_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2*
(MLP-logistic/camada_intermediaria_3/Tanh�
2MLP-logistic/camada_de_saida/MatMul/ReadVariableOpReadVariableOp;mlp_logistic_camada_de_saida_matmul_readvariableop_resource*
_output_shapes

:
*
dtype024
2MLP-logistic/camada_de_saida/MatMul/ReadVariableOp�
#MLP-logistic/camada_de_saida/MatMulMatMul,MLP-logistic/camada_intermediaria_3/Tanh:y:0:MLP-logistic/camada_de_saida/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2%
#MLP-logistic/camada_de_saida/MatMul�
3MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOpReadVariableOp<mlp_logistic_camada_de_saida_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOp�
$MLP-logistic/camada_de_saida/BiasAddBiasAdd-MLP-logistic/camada_de_saida/MatMul:product:0;MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2&
$MLP-logistic/camada_de_saida/BiasAdd�
IdentityIdentity-MLP-logistic/camada_de_saida/BiasAdd:output:06^MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOp5^MLP-logistic/camada_de_entrada/MatMul/ReadVariableOp4^MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOp3^MLP-logistic/camada_de_saida/MatMul/ReadVariableOp;^MLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOp:^MLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOp;^MLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOp:^MLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOp;^MLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOp:^MLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2n
5MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOp5MLP-logistic/camada_de_entrada/BiasAdd/ReadVariableOp2l
4MLP-logistic/camada_de_entrada/MatMul/ReadVariableOp4MLP-logistic/camada_de_entrada/MatMul/ReadVariableOp2j
3MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOp3MLP-logistic/camada_de_saida/BiasAdd/ReadVariableOp2h
2MLP-logistic/camada_de_saida/MatMul/ReadVariableOp2MLP-logistic/camada_de_saida/MatMul/ReadVariableOp2x
:MLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOp:MLP-logistic/camada_intermediaria_1/BiasAdd/ReadVariableOp2v
9MLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOp9MLP-logistic/camada_intermediaria_1/MatMul/ReadVariableOp2x
:MLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOp:MLP-logistic/camada_intermediaria_2/BiasAdd/ReadVariableOp2v
9MLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOp9MLP-logistic/camada_intermediaria_2/MatMul/ReadVariableOp2x
:MLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOp:MLP-logistic/camada_intermediaria_3/BiasAdd/ReadVariableOp2v
9MLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOp9MLP-logistic/camada_intermediaria_3/MatMul/ReadVariableOp:` \
'
_output_shapes
:���������
1
_user_specified_namecamada_de_entrada_input
�	
�
-__inference_MLP-logistic_layer_call_fn_367285
camada_de_entrada_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcamada_de_entrada_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_3672622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:���������
1
_user_specified_namecamada_de_entrada_input
�
�
-__inference_MLP-logistic_layer_call_fn_367498

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_3673162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367201
camada_de_entrada_input
camada_de_entrada_367088
camada_de_entrada_367090!
camada_intermediaria_1_367115!
camada_intermediaria_1_367117!
camada_intermediaria_2_367142!
camada_intermediaria_2_367144!
camada_intermediaria_3_367169!
camada_intermediaria_3_367171
camada_de_saida_367195
camada_de_saida_367197
identity��)camada_de_entrada/StatefulPartitionedCall�'camada_de_saida/StatefulPartitionedCall�.camada_intermediaria_1/StatefulPartitionedCall�.camada_intermediaria_2/StatefulPartitionedCall�.camada_intermediaria_3/StatefulPartitionedCall�
)camada_de_entrada/StatefulPartitionedCallStatefulPartitionedCallcamada_de_entrada_inputcamada_de_entrada_367088camada_de_entrada_367090*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_3670772+
)camada_de_entrada/StatefulPartitionedCall�
.camada_intermediaria_1/StatefulPartitionedCallStatefulPartitionedCall2camada_de_entrada/StatefulPartitionedCall:output:0camada_intermediaria_1_367115camada_intermediaria_1_367117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_36710420
.camada_intermediaria_1/StatefulPartitionedCall�
.camada_intermediaria_2/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_1/StatefulPartitionedCall:output:0camada_intermediaria_2_367142camada_intermediaria_2_367144*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_36713120
.camada_intermediaria_2/StatefulPartitionedCall�
.camada_intermediaria_3/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_2/StatefulPartitionedCall:output:0camada_intermediaria_3_367169camada_intermediaria_3_367171*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_36715820
.camada_intermediaria_3/StatefulPartitionedCall�
'camada_de_saida/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_3/StatefulPartitionedCall:output:0camada_de_saida_367195camada_de_saida_367197*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_3671842)
'camada_de_saida/StatefulPartitionedCall�
IdentityIdentity0camada_de_saida/StatefulPartitionedCall:output:0*^camada_de_entrada/StatefulPartitionedCall(^camada_de_saida/StatefulPartitionedCall/^camada_intermediaria_1/StatefulPartitionedCall/^camada_intermediaria_2/StatefulPartitionedCall/^camada_intermediaria_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2V
)camada_de_entrada/StatefulPartitionedCall)camada_de_entrada/StatefulPartitionedCall2R
'camada_de_saida/StatefulPartitionedCall'camada_de_saida/StatefulPartitionedCall2`
.camada_intermediaria_1/StatefulPartitionedCall.camada_intermediaria_1/StatefulPartitionedCall2`
.camada_intermediaria_2/StatefulPartitionedCall.camada_intermediaria_2/StatefulPartitionedCall2`
.camada_intermediaria_3/StatefulPartitionedCall.camada_intermediaria_3/StatefulPartitionedCall:` \
'
_output_shapes
:���������
1
_user_specified_namecamada_de_entrada_input
�
�
$__inference_signature_wrapper_367374
camada_de_entrada_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcamada_de_entrada_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_3670632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:���������
1
_user_specified_namecamada_de_entrada_input
�9
�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367411

inputs4
0camada_de_entrada_matmul_readvariableop_resource5
1camada_de_entrada_biasadd_readvariableop_resource9
5camada_intermediaria_1_matmul_readvariableop_resource:
6camada_intermediaria_1_biasadd_readvariableop_resource9
5camada_intermediaria_2_matmul_readvariableop_resource:
6camada_intermediaria_2_biasadd_readvariableop_resource9
5camada_intermediaria_3_matmul_readvariableop_resource:
6camada_intermediaria_3_biasadd_readvariableop_resource2
.camada_de_saida_matmul_readvariableop_resource3
/camada_de_saida_biasadd_readvariableop_resource
identity��(camada_de_entrada/BiasAdd/ReadVariableOp�'camada_de_entrada/MatMul/ReadVariableOp�&camada_de_saida/BiasAdd/ReadVariableOp�%camada_de_saida/MatMul/ReadVariableOp�-camada_intermediaria_1/BiasAdd/ReadVariableOp�,camada_intermediaria_1/MatMul/ReadVariableOp�-camada_intermediaria_2/BiasAdd/ReadVariableOp�,camada_intermediaria_2/MatMul/ReadVariableOp�-camada_intermediaria_3/BiasAdd/ReadVariableOp�,camada_intermediaria_3/MatMul/ReadVariableOp�
'camada_de_entrada/MatMul/ReadVariableOpReadVariableOp0camada_de_entrada_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'camada_de_entrada/MatMul/ReadVariableOp�
camada_de_entrada/MatMulMatMulinputs/camada_de_entrada/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_entrada/MatMul�
(camada_de_entrada/BiasAdd/ReadVariableOpReadVariableOp1camada_de_entrada_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(camada_de_entrada/BiasAdd/ReadVariableOp�
camada_de_entrada/BiasAddBiasAdd"camada_de_entrada/MatMul:product:00camada_de_entrada/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_entrada/BiasAdd�
,camada_intermediaria_1/MatMul/ReadVariableOpReadVariableOp5camada_intermediaria_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,camada_intermediaria_1/MatMul/ReadVariableOp�
camada_intermediaria_1/MatMulMatMul"camada_de_entrada/BiasAdd:output:04camada_intermediaria_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_1/MatMul�
-camada_intermediaria_1/BiasAdd/ReadVariableOpReadVariableOp6camada_intermediaria_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-camada_intermediaria_1/BiasAdd/ReadVariableOp�
camada_intermediaria_1/BiasAddBiasAdd'camada_intermediaria_1/MatMul:product:05camada_intermediaria_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
camada_intermediaria_1/BiasAdd�
camada_intermediaria_1/TanhTanh'camada_intermediaria_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_1/Tanh�
,camada_intermediaria_2/MatMul/ReadVariableOpReadVariableOp5camada_intermediaria_2_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02.
,camada_intermediaria_2/MatMul/ReadVariableOp�
camada_intermediaria_2/MatMulMatMulcamada_intermediaria_1/Tanh:y:04camada_intermediaria_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_2/MatMul�
-camada_intermediaria_2/BiasAdd/ReadVariableOpReadVariableOp6camada_intermediaria_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-camada_intermediaria_2/BiasAdd/ReadVariableOp�
camada_intermediaria_2/BiasAddBiasAdd'camada_intermediaria_2/MatMul:product:05camada_intermediaria_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
camada_intermediaria_2/BiasAdd�
camada_intermediaria_2/TanhTanh'camada_intermediaria_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_2/Tanh�
,camada_intermediaria_3/MatMul/ReadVariableOpReadVariableOp5camada_intermediaria_3_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02.
,camada_intermediaria_3/MatMul/ReadVariableOp�
camada_intermediaria_3/MatMulMatMulcamada_intermediaria_2/Tanh:y:04camada_intermediaria_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_3/MatMul�
-camada_intermediaria_3/BiasAdd/ReadVariableOpReadVariableOp6camada_intermediaria_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-camada_intermediaria_3/BiasAdd/ReadVariableOp�
camada_intermediaria_3/BiasAddBiasAdd'camada_intermediaria_3/MatMul:product:05camada_intermediaria_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
camada_intermediaria_3/BiasAdd�
camada_intermediaria_3/TanhTanh'camada_intermediaria_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_3/Tanh�
%camada_de_saida/MatMul/ReadVariableOpReadVariableOp.camada_de_saida_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%camada_de_saida/MatMul/ReadVariableOp�
camada_de_saida/MatMulMatMulcamada_intermediaria_3/Tanh:y:0-camada_de_saida/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_saida/MatMul�
&camada_de_saida/BiasAdd/ReadVariableOpReadVariableOp/camada_de_saida_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&camada_de_saida/BiasAdd/ReadVariableOp�
camada_de_saida/BiasAddBiasAdd camada_de_saida/MatMul:product:0.camada_de_saida/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_saida/BiasAdd�
IdentityIdentity camada_de_saida/BiasAdd:output:0)^camada_de_entrada/BiasAdd/ReadVariableOp(^camada_de_entrada/MatMul/ReadVariableOp'^camada_de_saida/BiasAdd/ReadVariableOp&^camada_de_saida/MatMul/ReadVariableOp.^camada_intermediaria_1/BiasAdd/ReadVariableOp-^camada_intermediaria_1/MatMul/ReadVariableOp.^camada_intermediaria_2/BiasAdd/ReadVariableOp-^camada_intermediaria_2/MatMul/ReadVariableOp.^camada_intermediaria_3/BiasAdd/ReadVariableOp-^camada_intermediaria_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2T
(camada_de_entrada/BiasAdd/ReadVariableOp(camada_de_entrada/BiasAdd/ReadVariableOp2R
'camada_de_entrada/MatMul/ReadVariableOp'camada_de_entrada/MatMul/ReadVariableOp2P
&camada_de_saida/BiasAdd/ReadVariableOp&camada_de_saida/BiasAdd/ReadVariableOp2N
%camada_de_saida/MatMul/ReadVariableOp%camada_de_saida/MatMul/ReadVariableOp2^
-camada_intermediaria_1/BiasAdd/ReadVariableOp-camada_intermediaria_1/BiasAdd/ReadVariableOp2\
,camada_intermediaria_1/MatMul/ReadVariableOp,camada_intermediaria_1/MatMul/ReadVariableOp2^
-camada_intermediaria_2/BiasAdd/ReadVariableOp-camada_intermediaria_2/BiasAdd/ReadVariableOp2\
,camada_intermediaria_2/MatMul/ReadVariableOp,camada_intermediaria_2/MatMul/ReadVariableOp2^
-camada_intermediaria_3/BiasAdd/ReadVariableOp-camada_intermediaria_3/BiasAdd/ReadVariableOp2\
,camada_intermediaria_3/MatMul/ReadVariableOp,camada_intermediaria_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_367587

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
� 
�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367262

inputs
camada_de_entrada_367236
camada_de_entrada_367238!
camada_intermediaria_1_367241!
camada_intermediaria_1_367243!
camada_intermediaria_2_367246!
camada_intermediaria_2_367248!
camada_intermediaria_3_367251!
camada_intermediaria_3_367253
camada_de_saida_367256
camada_de_saida_367258
identity��)camada_de_entrada/StatefulPartitionedCall�'camada_de_saida/StatefulPartitionedCall�.camada_intermediaria_1/StatefulPartitionedCall�.camada_intermediaria_2/StatefulPartitionedCall�.camada_intermediaria_3/StatefulPartitionedCall�
)camada_de_entrada/StatefulPartitionedCallStatefulPartitionedCallinputscamada_de_entrada_367236camada_de_entrada_367238*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_3670772+
)camada_de_entrada/StatefulPartitionedCall�
.camada_intermediaria_1/StatefulPartitionedCallStatefulPartitionedCall2camada_de_entrada/StatefulPartitionedCall:output:0camada_intermediaria_1_367241camada_intermediaria_1_367243*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_36710420
.camada_intermediaria_1/StatefulPartitionedCall�
.camada_intermediaria_2/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_1/StatefulPartitionedCall:output:0camada_intermediaria_2_367246camada_intermediaria_2_367248*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_36713120
.camada_intermediaria_2/StatefulPartitionedCall�
.camada_intermediaria_3/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_2/StatefulPartitionedCall:output:0camada_intermediaria_3_367251camada_intermediaria_3_367253*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_36715820
.camada_intermediaria_3/StatefulPartitionedCall�
'camada_de_saida/StatefulPartitionedCallStatefulPartitionedCall7camada_intermediaria_3/StatefulPartitionedCall:output:0camada_de_saida_367256camada_de_saida_367258*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_3671842)
'camada_de_saida/StatefulPartitionedCall�
IdentityIdentity0camada_de_saida/StatefulPartitionedCall:output:0*^camada_de_entrada/StatefulPartitionedCall(^camada_de_saida/StatefulPartitionedCall/^camada_intermediaria_1/StatefulPartitionedCall/^camada_intermediaria_2/StatefulPartitionedCall/^camada_intermediaria_3/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2V
)camada_de_entrada/StatefulPartitionedCall)camada_de_entrada/StatefulPartitionedCall2R
'camada_de_saida/StatefulPartitionedCall'camada_de_saida/StatefulPartitionedCall2`
.camada_intermediaria_1/StatefulPartitionedCall.camada_intermediaria_1/StatefulPartitionedCall2`
.camada_intermediaria_2/StatefulPartitionedCall.camada_intermediaria_2/StatefulPartitionedCall2`
.camada_intermediaria_3/StatefulPartitionedCall.camada_intermediaria_3/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_367104

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_367077

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�9
�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367448

inputs4
0camada_de_entrada_matmul_readvariableop_resource5
1camada_de_entrada_biasadd_readvariableop_resource9
5camada_intermediaria_1_matmul_readvariableop_resource:
6camada_intermediaria_1_biasadd_readvariableop_resource9
5camada_intermediaria_2_matmul_readvariableop_resource:
6camada_intermediaria_2_biasadd_readvariableop_resource9
5camada_intermediaria_3_matmul_readvariableop_resource:
6camada_intermediaria_3_biasadd_readvariableop_resource2
.camada_de_saida_matmul_readvariableop_resource3
/camada_de_saida_biasadd_readvariableop_resource
identity��(camada_de_entrada/BiasAdd/ReadVariableOp�'camada_de_entrada/MatMul/ReadVariableOp�&camada_de_saida/BiasAdd/ReadVariableOp�%camada_de_saida/MatMul/ReadVariableOp�-camada_intermediaria_1/BiasAdd/ReadVariableOp�,camada_intermediaria_1/MatMul/ReadVariableOp�-camada_intermediaria_2/BiasAdd/ReadVariableOp�,camada_intermediaria_2/MatMul/ReadVariableOp�-camada_intermediaria_3/BiasAdd/ReadVariableOp�,camada_intermediaria_3/MatMul/ReadVariableOp�
'camada_de_entrada/MatMul/ReadVariableOpReadVariableOp0camada_de_entrada_matmul_readvariableop_resource*
_output_shapes

:*
dtype02)
'camada_de_entrada/MatMul/ReadVariableOp�
camada_de_entrada/MatMulMatMulinputs/camada_de_entrada/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_entrada/MatMul�
(camada_de_entrada/BiasAdd/ReadVariableOpReadVariableOp1camada_de_entrada_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(camada_de_entrada/BiasAdd/ReadVariableOp�
camada_de_entrada/BiasAddBiasAdd"camada_de_entrada/MatMul:product:00camada_de_entrada/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_entrada/BiasAdd�
,camada_intermediaria_1/MatMul/ReadVariableOpReadVariableOp5camada_intermediaria_1_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02.
,camada_intermediaria_1/MatMul/ReadVariableOp�
camada_intermediaria_1/MatMulMatMul"camada_de_entrada/BiasAdd:output:04camada_intermediaria_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_1/MatMul�
-camada_intermediaria_1/BiasAdd/ReadVariableOpReadVariableOp6camada_intermediaria_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-camada_intermediaria_1/BiasAdd/ReadVariableOp�
camada_intermediaria_1/BiasAddBiasAdd'camada_intermediaria_1/MatMul:product:05camada_intermediaria_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
camada_intermediaria_1/BiasAdd�
camada_intermediaria_1/TanhTanh'camada_intermediaria_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_1/Tanh�
,camada_intermediaria_2/MatMul/ReadVariableOpReadVariableOp5camada_intermediaria_2_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02.
,camada_intermediaria_2/MatMul/ReadVariableOp�
camada_intermediaria_2/MatMulMatMulcamada_intermediaria_1/Tanh:y:04camada_intermediaria_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_2/MatMul�
-camada_intermediaria_2/BiasAdd/ReadVariableOpReadVariableOp6camada_intermediaria_2_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-camada_intermediaria_2/BiasAdd/ReadVariableOp�
camada_intermediaria_2/BiasAddBiasAdd'camada_intermediaria_2/MatMul:product:05camada_intermediaria_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
camada_intermediaria_2/BiasAdd�
camada_intermediaria_2/TanhTanh'camada_intermediaria_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_2/Tanh�
,camada_intermediaria_3/MatMul/ReadVariableOpReadVariableOp5camada_intermediaria_3_matmul_readvariableop_resource*
_output_shapes

:

*
dtype02.
,camada_intermediaria_3/MatMul/ReadVariableOp�
camada_intermediaria_3/MatMulMatMulcamada_intermediaria_2/Tanh:y:04camada_intermediaria_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_3/MatMul�
-camada_intermediaria_3/BiasAdd/ReadVariableOpReadVariableOp6camada_intermediaria_3_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-camada_intermediaria_3/BiasAdd/ReadVariableOp�
camada_intermediaria_3/BiasAddBiasAdd'camada_intermediaria_3/MatMul:product:05camada_intermediaria_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2 
camada_intermediaria_3/BiasAdd�
camada_intermediaria_3/TanhTanh'camada_intermediaria_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������
2
camada_intermediaria_3/Tanh�
%camada_de_saida/MatMul/ReadVariableOpReadVariableOp.camada_de_saida_matmul_readvariableop_resource*
_output_shapes

:
*
dtype02'
%camada_de_saida/MatMul/ReadVariableOp�
camada_de_saida/MatMulMatMulcamada_intermediaria_3/Tanh:y:0-camada_de_saida/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_saida/MatMul�
&camada_de_saida/BiasAdd/ReadVariableOpReadVariableOp/camada_de_saida_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&camada_de_saida/BiasAdd/ReadVariableOp�
camada_de_saida/BiasAddBiasAdd camada_de_saida/MatMul:product:0.camada_de_saida/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
camada_de_saida/BiasAdd�
IdentityIdentity camada_de_saida/BiasAdd:output:0)^camada_de_entrada/BiasAdd/ReadVariableOp(^camada_de_entrada/MatMul/ReadVariableOp'^camada_de_saida/BiasAdd/ReadVariableOp&^camada_de_saida/MatMul/ReadVariableOp.^camada_intermediaria_1/BiasAdd/ReadVariableOp-^camada_intermediaria_1/MatMul/ReadVariableOp.^camada_intermediaria_2/BiasAdd/ReadVariableOp-^camada_intermediaria_2/MatMul/ReadVariableOp.^camada_intermediaria_3/BiasAdd/ReadVariableOp-^camada_intermediaria_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::2T
(camada_de_entrada/BiasAdd/ReadVariableOp(camada_de_entrada/BiasAdd/ReadVariableOp2R
'camada_de_entrada/MatMul/ReadVariableOp'camada_de_entrada/MatMul/ReadVariableOp2P
&camada_de_saida/BiasAdd/ReadVariableOp&camada_de_saida/BiasAdd/ReadVariableOp2N
%camada_de_saida/MatMul/ReadVariableOp%camada_de_saida/MatMul/ReadVariableOp2^
-camada_intermediaria_1/BiasAdd/ReadVariableOp-camada_intermediaria_1/BiasAdd/ReadVariableOp2\
,camada_intermediaria_1/MatMul/ReadVariableOp,camada_intermediaria_1/MatMul/ReadVariableOp2^
-camada_intermediaria_2/BiasAdd/ReadVariableOp-camada_intermediaria_2/BiasAdd/ReadVariableOp2\
,camada_intermediaria_2/MatMul/ReadVariableOp,camada_intermediaria_2/MatMul/ReadVariableOp2^
-camada_intermediaria_3/BiasAdd/ReadVariableOp-camada_intermediaria_3/BiasAdd/ReadVariableOp2\
,camada_intermediaria_3/MatMul/ReadVariableOp,camada_intermediaria_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_367548

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:

*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������
2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�V
�
__inference__traced_save_367733
file_prefix7
3savev2_camada_de_entrada_kernel_read_readvariableop5
1savev2_camada_de_entrada_bias_read_readvariableop<
8savev2_camada_intermediaria_1_kernel_read_readvariableop:
6savev2_camada_intermediaria_1_bias_read_readvariableop<
8savev2_camada_intermediaria_2_kernel_read_readvariableop:
6savev2_camada_intermediaria_2_bias_read_readvariableop<
8savev2_camada_intermediaria_3_kernel_read_readvariableop:
6savev2_camada_intermediaria_3_bias_read_readvariableop5
1savev2_camada_de_saida_kernel_read_readvariableop3
/savev2_camada_de_saida_bias_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop)
%savev2_nadam_iter_read_readvariableop	3
/savev2_nadam_momentum_cache_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop?
;savev2_nadam_camada_de_entrada_kernel_m_read_readvariableop=
9savev2_nadam_camada_de_entrada_bias_m_read_readvariableopD
@savev2_nadam_camada_intermediaria_1_kernel_m_read_readvariableopB
>savev2_nadam_camada_intermediaria_1_bias_m_read_readvariableopD
@savev2_nadam_camada_intermediaria_2_kernel_m_read_readvariableopB
>savev2_nadam_camada_intermediaria_2_bias_m_read_readvariableopD
@savev2_nadam_camada_intermediaria_3_kernel_m_read_readvariableopB
>savev2_nadam_camada_intermediaria_3_bias_m_read_readvariableop=
9savev2_nadam_camada_de_saida_kernel_m_read_readvariableop;
7savev2_nadam_camada_de_saida_bias_m_read_readvariableop?
;savev2_nadam_camada_de_entrada_kernel_v_read_readvariableop=
9savev2_nadam_camada_de_entrada_bias_v_read_readvariableopD
@savev2_nadam_camada_intermediaria_1_kernel_v_read_readvariableopB
>savev2_nadam_camada_intermediaria_1_bias_v_read_readvariableopD
@savev2_nadam_camada_intermediaria_2_kernel_v_read_readvariableopB
>savev2_nadam_camada_intermediaria_2_bias_v_read_readvariableopD
@savev2_nadam_camada_intermediaria_3_kernel_v_read_readvariableopB
>savev2_nadam_camada_intermediaria_3_bias_v_read_readvariableop=
9savev2_nadam_camada_de_saida_kernel_v_read_readvariableop;
7savev2_nadam_camada_de_saida_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*�
value�B�'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/momentum_cache/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_camada_de_entrada_kernel_read_readvariableop1savev2_camada_de_entrada_bias_read_readvariableop8savev2_camada_intermediaria_1_kernel_read_readvariableop6savev2_camada_intermediaria_1_bias_read_readvariableop8savev2_camada_intermediaria_2_kernel_read_readvariableop6savev2_camada_intermediaria_2_bias_read_readvariableop8savev2_camada_intermediaria_3_kernel_read_readvariableop6savev2_camada_intermediaria_3_bias_read_readvariableop1savev2_camada_de_saida_kernel_read_readvariableop/savev2_camada_de_saida_bias_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop%savev2_nadam_iter_read_readvariableop/savev2_nadam_momentum_cache_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop;savev2_nadam_camada_de_entrada_kernel_m_read_readvariableop9savev2_nadam_camada_de_entrada_bias_m_read_readvariableop@savev2_nadam_camada_intermediaria_1_kernel_m_read_readvariableop>savev2_nadam_camada_intermediaria_1_bias_m_read_readvariableop@savev2_nadam_camada_intermediaria_2_kernel_m_read_readvariableop>savev2_nadam_camada_intermediaria_2_bias_m_read_readvariableop@savev2_nadam_camada_intermediaria_3_kernel_m_read_readvariableop>savev2_nadam_camada_intermediaria_3_bias_m_read_readvariableop9savev2_nadam_camada_de_saida_kernel_m_read_readvariableop7savev2_nadam_camada_de_saida_bias_m_read_readvariableop;savev2_nadam_camada_de_entrada_kernel_v_read_readvariableop9savev2_nadam_camada_de_entrada_bias_v_read_readvariableop@savev2_nadam_camada_intermediaria_1_kernel_v_read_readvariableop>savev2_nadam_camada_intermediaria_1_bias_v_read_readvariableop@savev2_nadam_camada_intermediaria_2_kernel_v_read_readvariableop>savev2_nadam_camada_intermediaria_2_bias_v_read_readvariableop@savev2_nadam_camada_intermediaria_3_kernel_v_read_readvariableop>savev2_nadam_camada_intermediaria_3_bias_v_read_readvariableop9savev2_nadam_camada_de_saida_kernel_v_read_readvariableop7savev2_nadam_camada_de_saida_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :::
:
:

:
:

:
:
:: : : : : : : : :::
:
:

:
:

:
:
::::
:
:

:
:

:
:
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$	 

_output_shapes

:
: 


_output_shapes
::
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:

: 

_output_shapes
:
:$ 

_output_shapes

:
: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:
:  

_output_shapes
:
:$! 

_output_shapes

:

: "

_output_shapes
:
:$# 

_output_shapes

:

: $

_output_shapes
:
:$% 

_output_shapes

:
: &

_output_shapes
::'

_output_shapes
: 
�
�
7__inference_camada_intermediaria_1_layer_call_fn_367537

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_3671042
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_camada_intermediaria_3_layer_call_fn_367577

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_3671582
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������
2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������

 
_user_specified_nameinputs
�	
�
-__inference_MLP-logistic_layer_call_fn_367339
camada_de_entrada_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallcamada_de_entrada_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_3673162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:���������::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:` \
'
_output_shapes
:���������
1
_user_specified_namecamada_de_entrada_input"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
[
camada_de_entrada_input@
)serving_default_camada_de_entrada_input:0���������C
camada_de_saida0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�1
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
regularization_losses
	variables
	trainable_variables

	keras_api

signatures
g_default_save_signature
h__call__
*i&call_and_return_all_conditional_losses"�.
_tf_keras_sequential�-{"class_name": "Sequential", "name": "MLP-logistic", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "MLP-logistic", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "camada_de_entrada_input"}}, {"class_name": "Dense", "config": {"name": "camada_de_entrada", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_intermediaria_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_intermediaria_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_intermediaria_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_de_saida", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "MLP-logistic", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "camada_de_entrada_input"}}, {"class_name": "Dense", "config": {"name": "camada_de_entrada", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_intermediaria_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_intermediaria_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_intermediaria_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "camada_de_saida", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Nadam", "config": {"name": "Nadam", "learning_rate": 0.003000000026077032, "decay": 0.004000000189989805, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07}}}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "camada_de_entrada", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "camada_de_entrada", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "camada_intermediaria_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "camada_intermediaria_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4]}}
�

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
n__call__
*o&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "camada_intermediaria_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "camada_intermediaria_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�

kernel
bias
 regularization_losses
!	variables
"trainable_variables
#	keras_api
p__call__
*q&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "camada_intermediaria_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "camada_intermediaria_3", "trainable": true, "dtype": "float32", "units": 10, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
r__call__
*s&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "camada_de_saida", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "camada_de_saida", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 10}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 10]}}
�

*beta_1

+beta_2
	,decay
-learning_rate
.iter
/momentum_cachemSmTmUmVmWmXmYmZ$m[%m\v]v^v_v`vavbvcvd$ve%vf"
	optimizer
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
$8
%9"
trackable_list_wrapper
�
regularization_losses
0layer_metrics

1layers
2non_trainable_variables
3metrics
4layer_regularization_losses
	variables
	trainable_variables
h__call__
g_default_save_signature
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
,
tserving_default"
signature_map
*:(2camada_de_entrada/kernel
$:"2camada_de_entrada/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
5layer_metrics

6layers
7non_trainable_variables
8metrics
9layer_regularization_losses
	variables
trainable_variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
/:-
2camada_intermediaria_1/kernel
):'
2camada_intermediaria_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
:layer_metrics

;layers
<non_trainable_variables
=metrics
>layer_regularization_losses
	variables
trainable_variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
/:-

2camada_intermediaria_2/kernel
):'
2camada_intermediaria_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses
?layer_metrics

@layers
Anon_trainable_variables
Bmetrics
Clayer_regularization_losses
	variables
trainable_variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
/:-

2camada_intermediaria_3/kernel
):'
2camada_intermediaria_3/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
 regularization_losses
Dlayer_metrics

Elayers
Fnon_trainable_variables
Gmetrics
Hlayer_regularization_losses
!	variables
"trainable_variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
(:&
2camada_de_saida/kernel
": 2camada_de_saida/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
&regularization_losses
Ilayer_metrics

Jlayers
Knon_trainable_variables
Lmetrics
Mlayer_regularization_losses
'	variables
(trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2
Nadam/iter
: (2Nadam/momentum_cache
 "
trackable_dict_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
'
N0"
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
 "
trackable_list_wrapper
�
	Ototal
	Pcount
Q	variables
R	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
O0
P1"
trackable_list_wrapper
-
Q	variables"
_generic_user_object
0:.2 Nadam/camada_de_entrada/kernel/m
*:(2Nadam/camada_de_entrada/bias/m
5:3
2%Nadam/camada_intermediaria_1/kernel/m
/:-
2#Nadam/camada_intermediaria_1/bias/m
5:3

2%Nadam/camada_intermediaria_2/kernel/m
/:-
2#Nadam/camada_intermediaria_2/bias/m
5:3

2%Nadam/camada_intermediaria_3/kernel/m
/:-
2#Nadam/camada_intermediaria_3/bias/m
.:,
2Nadam/camada_de_saida/kernel/m
(:&2Nadam/camada_de_saida/bias/m
0:.2 Nadam/camada_de_entrada/kernel/v
*:(2Nadam/camada_de_entrada/bias/v
5:3
2%Nadam/camada_intermediaria_1/kernel/v
/:-
2#Nadam/camada_intermediaria_1/bias/v
5:3

2%Nadam/camada_intermediaria_2/kernel/v
/:-
2#Nadam/camada_intermediaria_2/bias/v
5:3

2%Nadam/camada_intermediaria_3/kernel/v
/:-
2#Nadam/camada_intermediaria_3/bias/v
.:,
2Nadam/camada_de_saida/kernel/v
(:&2Nadam/camada_de_saida/bias/v
�2�
!__inference__wrapped_model_367063�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *6�3
1�.
camada_de_entrada_input���������
�2�
-__inference_MLP-logistic_layer_call_fn_367473
-__inference_MLP-logistic_layer_call_fn_367339
-__inference_MLP-logistic_layer_call_fn_367498
-__inference_MLP-logistic_layer_call_fn_367285�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367411
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367230
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367201
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367448�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_camada_de_entrada_layer_call_fn_367517�
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
�2�
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_367508�
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
�2�
7__inference_camada_intermediaria_1_layer_call_fn_367537�
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
�2�
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_367528�
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
�2�
7__inference_camada_intermediaria_2_layer_call_fn_367557�
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
�2�
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_367548�
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
�2�
7__inference_camada_intermediaria_3_layer_call_fn_367577�
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
�2�
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_367568�
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
�2�
0__inference_camada_de_saida_layer_call_fn_367596�
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
�2�
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_367587�
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
$__inference_signature_wrapper_367374camada_de_entrada_input"�
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
 �
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367201}
$%H�E
>�;
1�.
camada_de_entrada_input���������
p

 
� "%�"
�
0���������
� �
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367230}
$%H�E
>�;
1�.
camada_de_entrada_input���������
p 

 
� "%�"
�
0���������
� �
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367411l
$%7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
H__inference_MLP-logistic_layer_call_and_return_conditional_losses_367448l
$%7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
-__inference_MLP-logistic_layer_call_fn_367285p
$%H�E
>�;
1�.
camada_de_entrada_input���������
p

 
� "�����������
-__inference_MLP-logistic_layer_call_fn_367339p
$%H�E
>�;
1�.
camada_de_entrada_input���������
p 

 
� "�����������
-__inference_MLP-logistic_layer_call_fn_367473_
$%7�4
-�*
 �
inputs���������
p

 
� "�����������
-__inference_MLP-logistic_layer_call_fn_367498_
$%7�4
-�*
 �
inputs���������
p 

 
� "�����������
!__inference__wrapped_model_367063�
$%@�=
6�3
1�.
camada_de_entrada_input���������
� "A�>
<
camada_de_saida)�&
camada_de_saida����������
M__inference_camada_de_entrada_layer_call_and_return_conditional_losses_367508\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
2__inference_camada_de_entrada_layer_call_fn_367517O/�,
%�"
 �
inputs���������
� "�����������
K__inference_camada_de_saida_layer_call_and_return_conditional_losses_367587\$%/�,
%�"
 �
inputs���������

� "%�"
�
0���������
� �
0__inference_camada_de_saida_layer_call_fn_367596O$%/�,
%�"
 �
inputs���������

� "�����������
R__inference_camada_intermediaria_1_layer_call_and_return_conditional_losses_367528\/�,
%�"
 �
inputs���������
� "%�"
�
0���������

� �
7__inference_camada_intermediaria_1_layer_call_fn_367537O/�,
%�"
 �
inputs���������
� "����������
�
R__inference_camada_intermediaria_2_layer_call_and_return_conditional_losses_367548\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� �
7__inference_camada_intermediaria_2_layer_call_fn_367557O/�,
%�"
 �
inputs���������

� "����������
�
R__inference_camada_intermediaria_3_layer_call_and_return_conditional_losses_367568\/�,
%�"
 �
inputs���������

� "%�"
�
0���������

� �
7__inference_camada_intermediaria_3_layer_call_fn_367577O/�,
%�"
 �
inputs���������

� "����������
�
$__inference_signature_wrapper_367374�
$%[�X
� 
Q�N
L
camada_de_entrada_input1�.
camada_de_entrada_input���������"A�>
<
camada_de_saida)�&
camada_de_saida���������