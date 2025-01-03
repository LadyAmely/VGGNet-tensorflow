��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
�
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	"
grad_abool( "
grad_bbool( 
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
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
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �
9
VarIsInitializedOp
resource
is_initialized
�"serve*2.18.02v2.18.0-rc2-4-g6550e4bd8028ܹ
�
vgg_net/kernelVarHandleOp*
_output_shapes
: *

debug_namevgg_net/kernel/*
dtype0*
shape
:@
*
shared_namevgg_net/kernel
q
"vgg_net/kernel/Read/ReadVariableOpReadVariableOpvgg_net/kernel*
_output_shapes

:@
*
dtype0
�
vgg_net/kernel_1VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_1/*
dtype0*
shape:	�@*!
shared_namevgg_net/kernel_1
v
$vgg_net/kernel_1/Read/ReadVariableOpReadVariableOpvgg_net/kernel_1*
_output_shapes
:	�@*
dtype0
�
vgg_net/kernel_2VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_2/*
dtype0*
shape:
��*!
shared_namevgg_net/kernel_2
w
$vgg_net/kernel_2/Read/ReadVariableOpReadVariableOpvgg_net/kernel_2* 
_output_shapes
:
��*
dtype0
�
vgg_net/kernel_3VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_3/*
dtype0*
shape:��*!
shared_namevgg_net/kernel_3

$vgg_net/kernel_3/Read/ReadVariableOpReadVariableOpvgg_net/kernel_3*(
_output_shapes
:��*
dtype0
�
vgg_net/biasVarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias/*
dtype0*
shape:�*
shared_namevgg_net/bias
j
 vgg_net/bias/Read/ReadVariableOpReadVariableOpvgg_net/bias*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_4VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_4/*
dtype0*
shape:@�*!
shared_namevgg_net/kernel_4
~
$vgg_net/kernel_4/Read/ReadVariableOpReadVariableOpvgg_net/kernel_4*'
_output_shapes
:@�*
dtype0
�
vgg_net/kernel_5VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_5/*
dtype0*
shape: *!
shared_namevgg_net/kernel_5
}
$vgg_net/kernel_5/Read/ReadVariableOpReadVariableOpvgg_net/kernel_5*&
_output_shapes
: *
dtype0
�
vgg_net/bias_1VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_1/*
dtype0*
shape:@*
shared_namevgg_net/bias_1
m
"vgg_net/bias_1/Read/ReadVariableOpReadVariableOpvgg_net/bias_1*
_output_shapes
:@*
dtype0
�
vgg_net/bias_2VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_2/*
dtype0*
shape:�*
shared_namevgg_net/bias_2
n
"vgg_net/bias_2/Read/ReadVariableOpReadVariableOpvgg_net/bias_2*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_6VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_6/*
dtype0*
shape:  *!
shared_namevgg_net/kernel_6
}
$vgg_net/kernel_6/Read/ReadVariableOpReadVariableOpvgg_net/kernel_6*&
_output_shapes
:  *
dtype0
�
vgg_net/bias_3VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_3/*
dtype0*
shape: *
shared_namevgg_net/bias_3
m
"vgg_net/bias_3/Read/ReadVariableOpReadVariableOpvgg_net/bias_3*
_output_shapes
: *
dtype0
�
vgg_net/bias_4VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_4/*
dtype0*
shape:*
shared_namevgg_net/bias_4
m
"vgg_net/bias_4/Read/ReadVariableOpReadVariableOpvgg_net/bias_4*
_output_shapes
:*
dtype0
�
vgg_net/kernel_7VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_7/*
dtype0*
shape:��*!
shared_namevgg_net/kernel_7

$vgg_net/kernel_7/Read/ReadVariableOpReadVariableOpvgg_net/kernel_7*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_5VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_5/*
dtype0*
shape:�*
shared_namevgg_net/bias_5
n
"vgg_net/bias_5/Read/ReadVariableOpReadVariableOpvgg_net/bias_5*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_8VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_8/*
dtype0*
shape:��*!
shared_namevgg_net/kernel_8

$vgg_net/kernel_8/Read/ReadVariableOpReadVariableOpvgg_net/kernel_8*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_6VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_6/*
dtype0*
shape:@*
shared_namevgg_net/bias_6
m
"vgg_net/bias_6/Read/ReadVariableOpReadVariableOpvgg_net/bias_6*
_output_shapes
:@*
dtype0
�
vgg_net/kernel_9VarHandleOp*
_output_shapes
: *!

debug_namevgg_net/kernel_9/*
dtype0*
shape:@@*!
shared_namevgg_net/kernel_9
}
$vgg_net/kernel_9/Read/ReadVariableOpReadVariableOpvgg_net/kernel_9*&
_output_shapes
:@@*
dtype0
�
vgg_net/kernel_10VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_10/*
dtype0*
shape:*"
shared_namevgg_net/kernel_10

%vgg_net/kernel_10/Read/ReadVariableOpReadVariableOpvgg_net/kernel_10*&
_output_shapes
:*
dtype0
�
vgg_net/bias_7VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_7/*
dtype0*
shape:*
shared_namevgg_net/bias_7
m
"vgg_net/bias_7/Read/ReadVariableOpReadVariableOpvgg_net/bias_7*
_output_shapes
:*
dtype0
�
vgg_net/bias_8VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_8/*
dtype0*
shape:
*
shared_namevgg_net/bias_8
m
"vgg_net/bias_8/Read/ReadVariableOpReadVariableOpvgg_net/bias_8*
_output_shapes
:
*
dtype0
�
vgg_net/bias_9VarHandleOp*
_output_shapes
: *

debug_namevgg_net/bias_9/*
dtype0*
shape:@*
shared_namevgg_net/bias_9
m
"vgg_net/bias_9/Read/ReadVariableOpReadVariableOpvgg_net/bias_9*
_output_shapes
:@*
dtype0
�
vgg_net/bias_10VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_10/*
dtype0*
shape:�* 
shared_namevgg_net/bias_10
p
#vgg_net/bias_10/Read/ReadVariableOpReadVariableOpvgg_net/bias_10*
_output_shapes	
:�*
dtype0
�
vgg_net/bias_11VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_11/*
dtype0*
shape:�* 
shared_namevgg_net/bias_11
p
#vgg_net/bias_11/Read/ReadVariableOpReadVariableOpvgg_net/bias_11*
_output_shapes	
:�*
dtype0
�
vgg_net/bias_12VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_12/*
dtype0*
shape:�* 
shared_namevgg_net/bias_12
p
#vgg_net/bias_12/Read/ReadVariableOpReadVariableOpvgg_net/bias_12*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_11VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_11/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_11
�
%vgg_net/kernel_11/Read/ReadVariableOpReadVariableOpvgg_net/kernel_11*(
_output_shapes
:��*
dtype0
�
vgg_net/kernel_12VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_12/*
dtype0*
shape:@@*"
shared_namevgg_net/kernel_12

%vgg_net/kernel_12/Read/ReadVariableOpReadVariableOpvgg_net/kernel_12*&
_output_shapes
:@@*
dtype0
�
vgg_net/bias_13VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_13/*
dtype0*
shape:@* 
shared_namevgg_net/bias_13
o
#vgg_net/bias_13/Read/ReadVariableOpReadVariableOpvgg_net/bias_13*
_output_shapes
:@*
dtype0
�
vgg_net/kernel_13VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_13/*
dtype0*
shape:*"
shared_namevgg_net/kernel_13

%vgg_net/kernel_13/Read/ReadVariableOpReadVariableOpvgg_net/kernel_13*&
_output_shapes
:*
dtype0
�
vgg_net/kernel_14VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_14/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_14
�
%vgg_net/kernel_14/Read/ReadVariableOpReadVariableOpvgg_net/kernel_14*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_14VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_14/*
dtype0*
shape:�* 
shared_namevgg_net/bias_14
p
#vgg_net/bias_14/Read/ReadVariableOpReadVariableOpvgg_net/bias_14*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_15VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_15/*
dtype0*
shape: @*"
shared_namevgg_net/kernel_15

%vgg_net/kernel_15/Read/ReadVariableOpReadVariableOpvgg_net/kernel_15*&
_output_shapes
: @*
dtype0
�
vgg_net/bias_15VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_15/*
dtype0*
shape: * 
shared_namevgg_net/bias_15
o
#vgg_net/bias_15/Read/ReadVariableOpReadVariableOpvgg_net/bias_15*
_output_shapes
: *
dtype0
�
vgg_net/bias_16VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_16/*
dtype0*
shape:
* 
shared_namevgg_net/bias_16
o
#vgg_net/bias_16/Read/ReadVariableOpReadVariableOpvgg_net/bias_16*
_output_shapes
:
*
dtype0
�
#Variable/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_16*
_class
loc:@Variable*
_output_shapes
:
*
dtype0
�
VariableVarHandleOp*
_class
loc:@Variable*
_output_shapes
: *

debug_name	Variable/*
dtype0*
shape:
*
shared_name
Variable
a
)Variable/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable*
_output_shapes
: 
_
Variable/AssignAssignVariableOpVariable#Variable/Initializer/ReadVariableOp*
dtype0
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:
*
dtype0
�
vgg_net/kernel_16VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_16/*
dtype0*
shape
:@
*"
shared_namevgg_net/kernel_16
w
%vgg_net/kernel_16/Read/ReadVariableOpReadVariableOpvgg_net/kernel_16*
_output_shapes

:@
*
dtype0
�
%Variable_1/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_16*
_class
loc:@Variable_1*
_output_shapes

:@
*
dtype0
�

Variable_1VarHandleOp*
_class
loc:@Variable_1*
_output_shapes
: *

debug_nameVariable_1/*
dtype0*
shape
:@
*
shared_name
Variable_1
e
+Variable_1/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_1*
_output_shapes
: 
e
Variable_1/AssignAssignVariableOp
Variable_1%Variable_1/Initializer/ReadVariableOp*
dtype0
i
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes

:@
*
dtype0
�
%seed_generator_1/seed_generator_stateVarHandleOp*
_output_shapes
: *6

debug_name(&seed_generator_1/seed_generator_state/*
dtype0	*
shape:*6
shared_name'%seed_generator_1/seed_generator_state
�
9seed_generator_1/seed_generator_state/Read/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_2/Initializer/ReadVariableOpReadVariableOp%seed_generator_1/seed_generator_state*
_class
loc:@Variable_2*
_output_shapes
:*
dtype0	
�

Variable_2VarHandleOp*
_class
loc:@Variable_2*
_output_shapes
: *

debug_nameVariable_2/*
dtype0	*
shape:*
shared_name
Variable_2
e
+Variable_2/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_2*
_output_shapes
: 
e
Variable_2/AssignAssignVariableOp
Variable_2%Variable_2/Initializer/ReadVariableOp*
dtype0	
e
Variable_2/Read/ReadVariableOpReadVariableOp
Variable_2*
_output_shapes
:*
dtype0	
�
vgg_net/bias_17VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_17/*
dtype0*
shape:@* 
shared_namevgg_net/bias_17
o
#vgg_net/bias_17/Read/ReadVariableOpReadVariableOpvgg_net/bias_17*
_output_shapes
:@*
dtype0
�
%Variable_3/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_17*
_class
loc:@Variable_3*
_output_shapes
:@*
dtype0
�

Variable_3VarHandleOp*
_class
loc:@Variable_3*
_output_shapes
: *

debug_nameVariable_3/*
dtype0*
shape:@*
shared_name
Variable_3
e
+Variable_3/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_3*
_output_shapes
: 
e
Variable_3/AssignAssignVariableOp
Variable_3%Variable_3/Initializer/ReadVariableOp*
dtype0
e
Variable_3/Read/ReadVariableOpReadVariableOp
Variable_3*
_output_shapes
:@*
dtype0
�
vgg_net/kernel_17VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_17/*
dtype0*
shape:	�@*"
shared_namevgg_net/kernel_17
x
%vgg_net/kernel_17/Read/ReadVariableOpReadVariableOpvgg_net/kernel_17*
_output_shapes
:	�@*
dtype0
�
%Variable_4/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_17*
_class
loc:@Variable_4*
_output_shapes
:	�@*
dtype0
�

Variable_4VarHandleOp*
_class
loc:@Variable_4*
_output_shapes
: *

debug_nameVariable_4/*
dtype0*
shape:	�@*
shared_name
Variable_4
e
+Variable_4/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_4*
_output_shapes
: 
e
Variable_4/AssignAssignVariableOp
Variable_4%Variable_4/Initializer/ReadVariableOp*
dtype0
j
Variable_4/Read/ReadVariableOpReadVariableOp
Variable_4*
_output_shapes
:	�@*
dtype0
�
#seed_generator/seed_generator_stateVarHandleOp*
_output_shapes
: *4

debug_name&$seed_generator/seed_generator_state/*
dtype0	*
shape:*4
shared_name%#seed_generator/seed_generator_state
�
7seed_generator/seed_generator_state/Read/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_output_shapes
:*
dtype0	
�
%Variable_5/Initializer/ReadVariableOpReadVariableOp#seed_generator/seed_generator_state*
_class
loc:@Variable_5*
_output_shapes
:*
dtype0	
�

Variable_5VarHandleOp*
_class
loc:@Variable_5*
_output_shapes
: *

debug_nameVariable_5/*
dtype0	*
shape:*
shared_name
Variable_5
e
+Variable_5/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_5*
_output_shapes
: 
e
Variable_5/AssignAssignVariableOp
Variable_5%Variable_5/Initializer/ReadVariableOp*
dtype0	
e
Variable_5/Read/ReadVariableOpReadVariableOp
Variable_5*
_output_shapes
:*
dtype0	
�
vgg_net/bias_18VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_18/*
dtype0*
shape:�* 
shared_namevgg_net/bias_18
p
#vgg_net/bias_18/Read/ReadVariableOpReadVariableOpvgg_net/bias_18*
_output_shapes	
:�*
dtype0
�
%Variable_6/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_18*
_class
loc:@Variable_6*
_output_shapes	
:�*
dtype0
�

Variable_6VarHandleOp*
_class
loc:@Variable_6*
_output_shapes
: *

debug_nameVariable_6/*
dtype0*
shape:�*
shared_name
Variable_6
e
+Variable_6/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_6*
_output_shapes
: 
e
Variable_6/AssignAssignVariableOp
Variable_6%Variable_6/Initializer/ReadVariableOp*
dtype0
f
Variable_6/Read/ReadVariableOpReadVariableOp
Variable_6*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_18VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_18/*
dtype0*
shape:
��*"
shared_namevgg_net/kernel_18
y
%vgg_net/kernel_18/Read/ReadVariableOpReadVariableOpvgg_net/kernel_18* 
_output_shapes
:
��*
dtype0
�
%Variable_7/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_18*
_class
loc:@Variable_7* 
_output_shapes
:
��*
dtype0
�

Variable_7VarHandleOp*
_class
loc:@Variable_7*
_output_shapes
: *

debug_nameVariable_7/*
dtype0*
shape:
��*
shared_name
Variable_7
e
+Variable_7/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_7*
_output_shapes
: 
e
Variable_7/AssignAssignVariableOp
Variable_7%Variable_7/Initializer/ReadVariableOp*
dtype0
k
Variable_7/Read/ReadVariableOpReadVariableOp
Variable_7* 
_output_shapes
:
��*
dtype0
�
vgg_net/bias_19VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_19/*
dtype0*
shape:�* 
shared_namevgg_net/bias_19
p
#vgg_net/bias_19/Read/ReadVariableOpReadVariableOpvgg_net/bias_19*
_output_shapes	
:�*
dtype0
�
%Variable_8/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_19*
_class
loc:@Variable_8*
_output_shapes	
:�*
dtype0
�

Variable_8VarHandleOp*
_class
loc:@Variable_8*
_output_shapes
: *

debug_nameVariable_8/*
dtype0*
shape:�*
shared_name
Variable_8
e
+Variable_8/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_8*
_output_shapes
: 
e
Variable_8/AssignAssignVariableOp
Variable_8%Variable_8/Initializer/ReadVariableOp*
dtype0
f
Variable_8/Read/ReadVariableOpReadVariableOp
Variable_8*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_19VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_19/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_19
�
%vgg_net/kernel_19/Read/ReadVariableOpReadVariableOpvgg_net/kernel_19*(
_output_shapes
:��*
dtype0
�
%Variable_9/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_19*
_class
loc:@Variable_9*(
_output_shapes
:��*
dtype0
�

Variable_9VarHandleOp*
_class
loc:@Variable_9*
_output_shapes
: *

debug_nameVariable_9/*
dtype0*
shape:��*
shared_name
Variable_9
e
+Variable_9/IsInitialized/VarIsInitializedOpVarIsInitializedOp
Variable_9*
_output_shapes
: 
e
Variable_9/AssignAssignVariableOp
Variable_9%Variable_9/Initializer/ReadVariableOp*
dtype0
s
Variable_9/Read/ReadVariableOpReadVariableOp
Variable_9*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_20VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_20/*
dtype0*
shape:�* 
shared_namevgg_net/bias_20
p
#vgg_net/bias_20/Read/ReadVariableOpReadVariableOpvgg_net/bias_20*
_output_shapes	
:�*
dtype0
�
&Variable_10/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_20*
_class
loc:@Variable_10*
_output_shapes	
:�*
dtype0
�
Variable_10VarHandleOp*
_class
loc:@Variable_10*
_output_shapes
: *

debug_nameVariable_10/*
dtype0*
shape:�*
shared_nameVariable_10
g
,Variable_10/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_10*
_output_shapes
: 
h
Variable_10/AssignAssignVariableOpVariable_10&Variable_10/Initializer/ReadVariableOp*
dtype0
h
Variable_10/Read/ReadVariableOpReadVariableOpVariable_10*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_20VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_20/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_20
�
%vgg_net/kernel_20/Read/ReadVariableOpReadVariableOpvgg_net/kernel_20*(
_output_shapes
:��*
dtype0
�
&Variable_11/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_20*
_class
loc:@Variable_11*(
_output_shapes
:��*
dtype0
�
Variable_11VarHandleOp*
_class
loc:@Variable_11*
_output_shapes
: *

debug_nameVariable_11/*
dtype0*
shape:��*
shared_nameVariable_11
g
,Variable_11/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_11*
_output_shapes
: 
h
Variable_11/AssignAssignVariableOpVariable_11&Variable_11/Initializer/ReadVariableOp*
dtype0
u
Variable_11/Read/ReadVariableOpReadVariableOpVariable_11*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_21VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_21/*
dtype0*
shape:�* 
shared_namevgg_net/bias_21
p
#vgg_net/bias_21/Read/ReadVariableOpReadVariableOpvgg_net/bias_21*
_output_shapes	
:�*
dtype0
�
&Variable_12/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_21*
_class
loc:@Variable_12*
_output_shapes	
:�*
dtype0
�
Variable_12VarHandleOp*
_class
loc:@Variable_12*
_output_shapes
: *

debug_nameVariable_12/*
dtype0*
shape:�*
shared_nameVariable_12
g
,Variable_12/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_12*
_output_shapes
: 
h
Variable_12/AssignAssignVariableOpVariable_12&Variable_12/Initializer/ReadVariableOp*
dtype0
h
Variable_12/Read/ReadVariableOpReadVariableOpVariable_12*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_21VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_21/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_21
�
%vgg_net/kernel_21/Read/ReadVariableOpReadVariableOpvgg_net/kernel_21*(
_output_shapes
:��*
dtype0
�
&Variable_13/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_21*
_class
loc:@Variable_13*(
_output_shapes
:��*
dtype0
�
Variable_13VarHandleOp*
_class
loc:@Variable_13*
_output_shapes
: *

debug_nameVariable_13/*
dtype0*
shape:��*
shared_nameVariable_13
g
,Variable_13/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_13*
_output_shapes
: 
h
Variable_13/AssignAssignVariableOpVariable_13&Variable_13/Initializer/ReadVariableOp*
dtype0
u
Variable_13/Read/ReadVariableOpReadVariableOpVariable_13*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_22VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_22/*
dtype0*
shape:�* 
shared_namevgg_net/bias_22
p
#vgg_net/bias_22/Read/ReadVariableOpReadVariableOpvgg_net/bias_22*
_output_shapes	
:�*
dtype0
�
&Variable_14/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_22*
_class
loc:@Variable_14*
_output_shapes	
:�*
dtype0
�
Variable_14VarHandleOp*
_class
loc:@Variable_14*
_output_shapes
: *

debug_nameVariable_14/*
dtype0*
shape:�*
shared_nameVariable_14
g
,Variable_14/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_14*
_output_shapes
: 
h
Variable_14/AssignAssignVariableOpVariable_14&Variable_14/Initializer/ReadVariableOp*
dtype0
h
Variable_14/Read/ReadVariableOpReadVariableOpVariable_14*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_22VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_22/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_22
�
%vgg_net/kernel_22/Read/ReadVariableOpReadVariableOpvgg_net/kernel_22*(
_output_shapes
:��*
dtype0
�
&Variable_15/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_22*
_class
loc:@Variable_15*(
_output_shapes
:��*
dtype0
�
Variable_15VarHandleOp*
_class
loc:@Variable_15*
_output_shapes
: *

debug_nameVariable_15/*
dtype0*
shape:��*
shared_nameVariable_15
g
,Variable_15/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_15*
_output_shapes
: 
h
Variable_15/AssignAssignVariableOpVariable_15&Variable_15/Initializer/ReadVariableOp*
dtype0
u
Variable_15/Read/ReadVariableOpReadVariableOpVariable_15*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_23VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_23/*
dtype0*
shape:�* 
shared_namevgg_net/bias_23
p
#vgg_net/bias_23/Read/ReadVariableOpReadVariableOpvgg_net/bias_23*
_output_shapes	
:�*
dtype0
�
&Variable_16/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_23*
_class
loc:@Variable_16*
_output_shapes	
:�*
dtype0
�
Variable_16VarHandleOp*
_class
loc:@Variable_16*
_output_shapes
: *

debug_nameVariable_16/*
dtype0*
shape:�*
shared_nameVariable_16
g
,Variable_16/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_16*
_output_shapes
: 
h
Variable_16/AssignAssignVariableOpVariable_16&Variable_16/Initializer/ReadVariableOp*
dtype0
h
Variable_16/Read/ReadVariableOpReadVariableOpVariable_16*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_23VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_23/*
dtype0*
shape:��*"
shared_namevgg_net/kernel_23
�
%vgg_net/kernel_23/Read/ReadVariableOpReadVariableOpvgg_net/kernel_23*(
_output_shapes
:��*
dtype0
�
&Variable_17/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_23*
_class
loc:@Variable_17*(
_output_shapes
:��*
dtype0
�
Variable_17VarHandleOp*
_class
loc:@Variable_17*
_output_shapes
: *

debug_nameVariable_17/*
dtype0*
shape:��*
shared_nameVariable_17
g
,Variable_17/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_17*
_output_shapes
: 
h
Variable_17/AssignAssignVariableOpVariable_17&Variable_17/Initializer/ReadVariableOp*
dtype0
u
Variable_17/Read/ReadVariableOpReadVariableOpVariable_17*(
_output_shapes
:��*
dtype0
�
vgg_net/bias_24VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_24/*
dtype0*
shape:�* 
shared_namevgg_net/bias_24
p
#vgg_net/bias_24/Read/ReadVariableOpReadVariableOpvgg_net/bias_24*
_output_shapes	
:�*
dtype0
�
&Variable_18/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_24*
_class
loc:@Variable_18*
_output_shapes	
:�*
dtype0
�
Variable_18VarHandleOp*
_class
loc:@Variable_18*
_output_shapes
: *

debug_nameVariable_18/*
dtype0*
shape:�*
shared_nameVariable_18
g
,Variable_18/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_18*
_output_shapes
: 
h
Variable_18/AssignAssignVariableOpVariable_18&Variable_18/Initializer/ReadVariableOp*
dtype0
h
Variable_18/Read/ReadVariableOpReadVariableOpVariable_18*
_output_shapes	
:�*
dtype0
�
vgg_net/kernel_24VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_24/*
dtype0*
shape:@�*"
shared_namevgg_net/kernel_24
�
%vgg_net/kernel_24/Read/ReadVariableOpReadVariableOpvgg_net/kernel_24*'
_output_shapes
:@�*
dtype0
�
&Variable_19/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_24*
_class
loc:@Variable_19*'
_output_shapes
:@�*
dtype0
�
Variable_19VarHandleOp*
_class
loc:@Variable_19*
_output_shapes
: *

debug_nameVariable_19/*
dtype0*
shape:@�*
shared_nameVariable_19
g
,Variable_19/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_19*
_output_shapes
: 
h
Variable_19/AssignAssignVariableOpVariable_19&Variable_19/Initializer/ReadVariableOp*
dtype0
t
Variable_19/Read/ReadVariableOpReadVariableOpVariable_19*'
_output_shapes
:@�*
dtype0
�
vgg_net/bias_25VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_25/*
dtype0*
shape:@* 
shared_namevgg_net/bias_25
o
#vgg_net/bias_25/Read/ReadVariableOpReadVariableOpvgg_net/bias_25*
_output_shapes
:@*
dtype0
�
&Variable_20/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_25*
_class
loc:@Variable_20*
_output_shapes
:@*
dtype0
�
Variable_20VarHandleOp*
_class
loc:@Variable_20*
_output_shapes
: *

debug_nameVariable_20/*
dtype0*
shape:@*
shared_nameVariable_20
g
,Variable_20/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_20*
_output_shapes
: 
h
Variable_20/AssignAssignVariableOpVariable_20&Variable_20/Initializer/ReadVariableOp*
dtype0
g
Variable_20/Read/ReadVariableOpReadVariableOpVariable_20*
_output_shapes
:@*
dtype0
�
vgg_net/kernel_25VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_25/*
dtype0*
shape:@@*"
shared_namevgg_net/kernel_25

%vgg_net/kernel_25/Read/ReadVariableOpReadVariableOpvgg_net/kernel_25*&
_output_shapes
:@@*
dtype0
�
&Variable_21/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_25*
_class
loc:@Variable_21*&
_output_shapes
:@@*
dtype0
�
Variable_21VarHandleOp*
_class
loc:@Variable_21*
_output_shapes
: *

debug_nameVariable_21/*
dtype0*
shape:@@*
shared_nameVariable_21
g
,Variable_21/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_21*
_output_shapes
: 
h
Variable_21/AssignAssignVariableOpVariable_21&Variable_21/Initializer/ReadVariableOp*
dtype0
s
Variable_21/Read/ReadVariableOpReadVariableOpVariable_21*&
_output_shapes
:@@*
dtype0
�
vgg_net/bias_26VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_26/*
dtype0*
shape:@* 
shared_namevgg_net/bias_26
o
#vgg_net/bias_26/Read/ReadVariableOpReadVariableOpvgg_net/bias_26*
_output_shapes
:@*
dtype0
�
&Variable_22/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_26*
_class
loc:@Variable_22*
_output_shapes
:@*
dtype0
�
Variable_22VarHandleOp*
_class
loc:@Variable_22*
_output_shapes
: *

debug_nameVariable_22/*
dtype0*
shape:@*
shared_nameVariable_22
g
,Variable_22/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_22*
_output_shapes
: 
h
Variable_22/AssignAssignVariableOpVariable_22&Variable_22/Initializer/ReadVariableOp*
dtype0
g
Variable_22/Read/ReadVariableOpReadVariableOpVariable_22*
_output_shapes
:@*
dtype0
�
vgg_net/kernel_26VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_26/*
dtype0*
shape:@@*"
shared_namevgg_net/kernel_26

%vgg_net/kernel_26/Read/ReadVariableOpReadVariableOpvgg_net/kernel_26*&
_output_shapes
:@@*
dtype0
�
&Variable_23/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_26*
_class
loc:@Variable_23*&
_output_shapes
:@@*
dtype0
�
Variable_23VarHandleOp*
_class
loc:@Variable_23*
_output_shapes
: *

debug_nameVariable_23/*
dtype0*
shape:@@*
shared_nameVariable_23
g
,Variable_23/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_23*
_output_shapes
: 
h
Variable_23/AssignAssignVariableOpVariable_23&Variable_23/Initializer/ReadVariableOp*
dtype0
s
Variable_23/Read/ReadVariableOpReadVariableOpVariable_23*&
_output_shapes
:@@*
dtype0
�
vgg_net/bias_27VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_27/*
dtype0*
shape:@* 
shared_namevgg_net/bias_27
o
#vgg_net/bias_27/Read/ReadVariableOpReadVariableOpvgg_net/bias_27*
_output_shapes
:@*
dtype0
�
&Variable_24/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_27*
_class
loc:@Variable_24*
_output_shapes
:@*
dtype0
�
Variable_24VarHandleOp*
_class
loc:@Variable_24*
_output_shapes
: *

debug_nameVariable_24/*
dtype0*
shape:@*
shared_nameVariable_24
g
,Variable_24/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_24*
_output_shapes
: 
h
Variable_24/AssignAssignVariableOpVariable_24&Variable_24/Initializer/ReadVariableOp*
dtype0
g
Variable_24/Read/ReadVariableOpReadVariableOpVariable_24*
_output_shapes
:@*
dtype0
�
vgg_net/kernel_27VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_27/*
dtype0*
shape: @*"
shared_namevgg_net/kernel_27

%vgg_net/kernel_27/Read/ReadVariableOpReadVariableOpvgg_net/kernel_27*&
_output_shapes
: @*
dtype0
�
&Variable_25/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_27*
_class
loc:@Variable_25*&
_output_shapes
: @*
dtype0
�
Variable_25VarHandleOp*
_class
loc:@Variable_25*
_output_shapes
: *

debug_nameVariable_25/*
dtype0*
shape: @*
shared_nameVariable_25
g
,Variable_25/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_25*
_output_shapes
: 
h
Variable_25/AssignAssignVariableOpVariable_25&Variable_25/Initializer/ReadVariableOp*
dtype0
s
Variable_25/Read/ReadVariableOpReadVariableOpVariable_25*&
_output_shapes
: @*
dtype0
�
vgg_net/bias_28VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_28/*
dtype0*
shape: * 
shared_namevgg_net/bias_28
o
#vgg_net/bias_28/Read/ReadVariableOpReadVariableOpvgg_net/bias_28*
_output_shapes
: *
dtype0
�
&Variable_26/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_28*
_class
loc:@Variable_26*
_output_shapes
: *
dtype0
�
Variable_26VarHandleOp*
_class
loc:@Variable_26*
_output_shapes
: *

debug_nameVariable_26/*
dtype0*
shape: *
shared_nameVariable_26
g
,Variable_26/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_26*
_output_shapes
: 
h
Variable_26/AssignAssignVariableOpVariable_26&Variable_26/Initializer/ReadVariableOp*
dtype0
g
Variable_26/Read/ReadVariableOpReadVariableOpVariable_26*
_output_shapes
: *
dtype0
�
vgg_net/kernel_28VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_28/*
dtype0*
shape:  *"
shared_namevgg_net/kernel_28

%vgg_net/kernel_28/Read/ReadVariableOpReadVariableOpvgg_net/kernel_28*&
_output_shapes
:  *
dtype0
�
&Variable_27/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_28*
_class
loc:@Variable_27*&
_output_shapes
:  *
dtype0
�
Variable_27VarHandleOp*
_class
loc:@Variable_27*
_output_shapes
: *

debug_nameVariable_27/*
dtype0*
shape:  *
shared_nameVariable_27
g
,Variable_27/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_27*
_output_shapes
: 
h
Variable_27/AssignAssignVariableOpVariable_27&Variable_27/Initializer/ReadVariableOp*
dtype0
s
Variable_27/Read/ReadVariableOpReadVariableOpVariable_27*&
_output_shapes
:  *
dtype0
�
vgg_net/bias_29VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_29/*
dtype0*
shape: * 
shared_namevgg_net/bias_29
o
#vgg_net/bias_29/Read/ReadVariableOpReadVariableOpvgg_net/bias_29*
_output_shapes
: *
dtype0
�
&Variable_28/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_29*
_class
loc:@Variable_28*
_output_shapes
: *
dtype0
�
Variable_28VarHandleOp*
_class
loc:@Variable_28*
_output_shapes
: *

debug_nameVariable_28/*
dtype0*
shape: *
shared_nameVariable_28
g
,Variable_28/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_28*
_output_shapes
: 
h
Variable_28/AssignAssignVariableOpVariable_28&Variable_28/Initializer/ReadVariableOp*
dtype0
g
Variable_28/Read/ReadVariableOpReadVariableOpVariable_28*
_output_shapes
: *
dtype0
�
vgg_net/kernel_29VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_29/*
dtype0*
shape: *"
shared_namevgg_net/kernel_29

%vgg_net/kernel_29/Read/ReadVariableOpReadVariableOpvgg_net/kernel_29*&
_output_shapes
: *
dtype0
�
&Variable_29/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_29*
_class
loc:@Variable_29*&
_output_shapes
: *
dtype0
�
Variable_29VarHandleOp*
_class
loc:@Variable_29*
_output_shapes
: *

debug_nameVariable_29/*
dtype0*
shape: *
shared_nameVariable_29
g
,Variable_29/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_29*
_output_shapes
: 
h
Variable_29/AssignAssignVariableOpVariable_29&Variable_29/Initializer/ReadVariableOp*
dtype0
s
Variable_29/Read/ReadVariableOpReadVariableOpVariable_29*&
_output_shapes
: *
dtype0
�
vgg_net/bias_30VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_30/*
dtype0*
shape:* 
shared_namevgg_net/bias_30
o
#vgg_net/bias_30/Read/ReadVariableOpReadVariableOpvgg_net/bias_30*
_output_shapes
:*
dtype0
�
&Variable_30/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_30*
_class
loc:@Variable_30*
_output_shapes
:*
dtype0
�
Variable_30VarHandleOp*
_class
loc:@Variable_30*
_output_shapes
: *

debug_nameVariable_30/*
dtype0*
shape:*
shared_nameVariable_30
g
,Variable_30/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_30*
_output_shapes
: 
h
Variable_30/AssignAssignVariableOpVariable_30&Variable_30/Initializer/ReadVariableOp*
dtype0
g
Variable_30/Read/ReadVariableOpReadVariableOpVariable_30*
_output_shapes
:*
dtype0
�
vgg_net/kernel_30VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_30/*
dtype0*
shape:*"
shared_namevgg_net/kernel_30

%vgg_net/kernel_30/Read/ReadVariableOpReadVariableOpvgg_net/kernel_30*&
_output_shapes
:*
dtype0
�
&Variable_31/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_30*
_class
loc:@Variable_31*&
_output_shapes
:*
dtype0
�
Variable_31VarHandleOp*
_class
loc:@Variable_31*
_output_shapes
: *

debug_nameVariable_31/*
dtype0*
shape:*
shared_nameVariable_31
g
,Variable_31/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_31*
_output_shapes
: 
h
Variable_31/AssignAssignVariableOpVariable_31&Variable_31/Initializer/ReadVariableOp*
dtype0
s
Variable_31/Read/ReadVariableOpReadVariableOpVariable_31*&
_output_shapes
:*
dtype0
�
vgg_net/bias_31VarHandleOp*
_output_shapes
: * 

debug_namevgg_net/bias_31/*
dtype0*
shape:* 
shared_namevgg_net/bias_31
o
#vgg_net/bias_31/Read/ReadVariableOpReadVariableOpvgg_net/bias_31*
_output_shapes
:*
dtype0
�
&Variable_32/Initializer/ReadVariableOpReadVariableOpvgg_net/bias_31*
_class
loc:@Variable_32*
_output_shapes
:*
dtype0
�
Variable_32VarHandleOp*
_class
loc:@Variable_32*
_output_shapes
: *

debug_nameVariable_32/*
dtype0*
shape:*
shared_nameVariable_32
g
,Variable_32/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_32*
_output_shapes
: 
h
Variable_32/AssignAssignVariableOpVariable_32&Variable_32/Initializer/ReadVariableOp*
dtype0
g
Variable_32/Read/ReadVariableOpReadVariableOpVariable_32*
_output_shapes
:*
dtype0
�
vgg_net/kernel_31VarHandleOp*
_output_shapes
: *"

debug_namevgg_net/kernel_31/*
dtype0*
shape:*"
shared_namevgg_net/kernel_31

%vgg_net/kernel_31/Read/ReadVariableOpReadVariableOpvgg_net/kernel_31*&
_output_shapes
:*
dtype0
�
&Variable_33/Initializer/ReadVariableOpReadVariableOpvgg_net/kernel_31*
_class
loc:@Variable_33*&
_output_shapes
:*
dtype0
�
Variable_33VarHandleOp*
_class
loc:@Variable_33*
_output_shapes
: *

debug_nameVariable_33/*
dtype0*
shape:*
shared_nameVariable_33
g
,Variable_33/IsInitialized/VarIsInitializedOpVarIsInitializedOpVariable_33*
_output_shapes
: 
h
Variable_33/AssignAssignVariableOpVariable_33&Variable_33/Initializer/ReadVariableOp*
dtype0
s
Variable_33/Read/ReadVariableOpReadVariableOpVariable_33*&
_output_shapes
:*
dtype0

serve_args_0Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserve_args_0vgg_net/kernel_31vgg_net/bias_31vgg_net/kernel_30vgg_net/bias_30vgg_net/kernel_29vgg_net/bias_29vgg_net/kernel_28vgg_net/bias_28vgg_net/kernel_27vgg_net/bias_27vgg_net/kernel_26vgg_net/bias_26vgg_net/kernel_25vgg_net/bias_25vgg_net/kernel_24vgg_net/bias_24vgg_net/kernel_23vgg_net/bias_23vgg_net/kernel_22vgg_net/bias_22vgg_net/kernel_21vgg_net/bias_21vgg_net/kernel_20vgg_net/bias_20vgg_net/kernel_19vgg_net/bias_19vgg_net/kernel_18vgg_net/bias_18vgg_net/kernel_17vgg_net/bias_17vgg_net/kernel_16vgg_net/bias_16*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU 2J 8� �J *6
f1R/
-__inference_signature_wrapper___call___129836
�
serving_default_args_0Placeholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCall_1StatefulPartitionedCallserving_default_args_0vgg_net/kernel_31vgg_net/bias_31vgg_net/kernel_30vgg_net/bias_30vgg_net/kernel_29vgg_net/bias_29vgg_net/kernel_28vgg_net/bias_28vgg_net/kernel_27vgg_net/bias_27vgg_net/kernel_26vgg_net/bias_26vgg_net/kernel_25vgg_net/bias_25vgg_net/kernel_24vgg_net/bias_24vgg_net/kernel_23vgg_net/bias_23vgg_net/kernel_22vgg_net/bias_22vgg_net/kernel_21vgg_net/bias_21vgg_net/kernel_20vgg_net/bias_20vgg_net/kernel_19vgg_net/bias_19vgg_net/kernel_18vgg_net/bias_18vgg_net/kernel_17vgg_net/bias_17vgg_net/kernel_16vgg_net/bias_16*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU 2J 8� �J *6
f1R/
-__inference_signature_wrapper___call___129905

NoOpNoOp
�3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�2
value�2B�2 B�2
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33*
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
%28
&29
(30
)31*

$0
'1*
�
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
<18
=19
>20
?21
@22
A23
B24
C25
D26
E27
F28
G29
H30
I31*
* 

Jtrace_0* 
"
	Kserve
Lserving_default* 
KE
VARIABLE_VALUEVariable_33&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_32&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_31&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_30&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_29&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_28&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_27&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_26&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_25&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEVariable_24&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_23'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_22'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_21'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_20'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_19'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_18'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_17'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_16'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_15'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_14'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_13'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_12'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_11'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
LF
VARIABLE_VALUEVariable_10'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_9'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_8'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_7'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_6'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_5'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_4'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_3'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_2'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
Variable_1'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEVariable'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEvgg_net/bias_28+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEvgg_net/kernel_27+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEvgg_net/bias_21+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEvgg_net/kernel_20+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEvgg_net/kernel_31+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEvgg_net/bias_27+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEvgg_net/kernel_26+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEvgg_net/kernel_23+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEvgg_net/bias_20+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEvgg_net/bias_19+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_18,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_17,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_16,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_31,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_30,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_25,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_26,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_19,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_23,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_22,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_30,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_29,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_28,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_24,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_25,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_29,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_24,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEvgg_net/bias_22,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_21,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_18,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_17,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEvgg_net/kernel_16,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablevgg_net/bias_28vgg_net/kernel_27vgg_net/bias_21vgg_net/kernel_20vgg_net/kernel_31vgg_net/bias_27vgg_net/kernel_26vgg_net/kernel_23vgg_net/bias_20vgg_net/bias_19vgg_net/bias_18vgg_net/bias_17vgg_net/bias_16vgg_net/bias_31vgg_net/kernel_30vgg_net/kernel_25vgg_net/bias_26vgg_net/kernel_19vgg_net/bias_23vgg_net/kernel_22vgg_net/bias_30vgg_net/bias_29vgg_net/kernel_28vgg_net/bias_24vgg_net/bias_25vgg_net/kernel_29vgg_net/kernel_24vgg_net/bias_22vgg_net/kernel_21vgg_net/kernel_18vgg_net/kernel_17vgg_net/kernel_16Const*O
TinH
F2D*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *(
f#R!
__inference__traced_save_130461
�

StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable_33Variable_32Variable_31Variable_30Variable_29Variable_28Variable_27Variable_26Variable_25Variable_24Variable_23Variable_22Variable_21Variable_20Variable_19Variable_18Variable_17Variable_16Variable_15Variable_14Variable_13Variable_12Variable_11Variable_10
Variable_9
Variable_8
Variable_7
Variable_6
Variable_5
Variable_4
Variable_3
Variable_2
Variable_1Variablevgg_net/bias_28vgg_net/kernel_27vgg_net/bias_21vgg_net/kernel_20vgg_net/kernel_31vgg_net/bias_27vgg_net/kernel_26vgg_net/kernel_23vgg_net/bias_20vgg_net/bias_19vgg_net/bias_18vgg_net/bias_17vgg_net/bias_16vgg_net/bias_31vgg_net/kernel_30vgg_net/kernel_25vgg_net/bias_26vgg_net/kernel_19vgg_net/bias_23vgg_net/kernel_22vgg_net/bias_30vgg_net/bias_29vgg_net/kernel_28vgg_net/bias_24vgg_net/bias_25vgg_net/kernel_29vgg_net/kernel_24vgg_net/bias_22vgg_net/kernel_21vgg_net/kernel_18vgg_net/kernel_17vgg_net/kernel_16*N
TinG
E2C*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU 2J 8� �J *+
f&R$
"__inference__traced_restore_130668��
Ԣ
�(
"__inference__traced_restore_130668
file_prefix6
assignvariableop_variable_33:,
assignvariableop_1_variable_32:8
assignvariableop_2_variable_31:,
assignvariableop_3_variable_30:8
assignvariableop_4_variable_29: ,
assignvariableop_5_variable_28: 8
assignvariableop_6_variable_27:  ,
assignvariableop_7_variable_26: 8
assignvariableop_8_variable_25: @,
assignvariableop_9_variable_24:@9
assignvariableop_10_variable_23:@@-
assignvariableop_11_variable_22:@9
assignvariableop_12_variable_21:@@-
assignvariableop_13_variable_20:@:
assignvariableop_14_variable_19:@�.
assignvariableop_15_variable_18:	�;
assignvariableop_16_variable_17:��.
assignvariableop_17_variable_16:	�;
assignvariableop_18_variable_15:��.
assignvariableop_19_variable_14:	�;
assignvariableop_20_variable_13:��.
assignvariableop_21_variable_12:	�;
assignvariableop_22_variable_11:��.
assignvariableop_23_variable_10:	�:
assignvariableop_24_variable_9:��-
assignvariableop_25_variable_8:	�2
assignvariableop_26_variable_7:
��-
assignvariableop_27_variable_6:	�,
assignvariableop_28_variable_5:	1
assignvariableop_29_variable_4:	�@,
assignvariableop_30_variable_3:@,
assignvariableop_31_variable_2:	0
assignvariableop_32_variable_1:@
*
assignvariableop_33_variable:
1
#assignvariableop_34_vgg_net_bias_28: ?
%assignvariableop_35_vgg_net_kernel_27: @2
#assignvariableop_36_vgg_net_bias_21:	�A
%assignvariableop_37_vgg_net_kernel_20:��?
%assignvariableop_38_vgg_net_kernel_31:1
#assignvariableop_39_vgg_net_bias_27:@?
%assignvariableop_40_vgg_net_kernel_26:@@A
%assignvariableop_41_vgg_net_kernel_23:��2
#assignvariableop_42_vgg_net_bias_20:	�2
#assignvariableop_43_vgg_net_bias_19:	�2
#assignvariableop_44_vgg_net_bias_18:	�1
#assignvariableop_45_vgg_net_bias_17:@1
#assignvariableop_46_vgg_net_bias_16:
1
#assignvariableop_47_vgg_net_bias_31:?
%assignvariableop_48_vgg_net_kernel_30:?
%assignvariableop_49_vgg_net_kernel_25:@@1
#assignvariableop_50_vgg_net_bias_26:@A
%assignvariableop_51_vgg_net_kernel_19:��2
#assignvariableop_52_vgg_net_bias_23:	�A
%assignvariableop_53_vgg_net_kernel_22:��1
#assignvariableop_54_vgg_net_bias_30:1
#assignvariableop_55_vgg_net_bias_29: ?
%assignvariableop_56_vgg_net_kernel_28:  2
#assignvariableop_57_vgg_net_bias_24:	�1
#assignvariableop_58_vgg_net_bias_25:@?
%assignvariableop_59_vgg_net_kernel_29: @
%assignvariableop_60_vgg_net_kernel_24:@�2
#assignvariableop_61_vgg_net_bias_22:	�A
%assignvariableop_62_vgg_net_kernel_21:��9
%assignvariableop_63_vgg_net_kernel_18:
��8
%assignvariableop_64_vgg_net_kernel_17:	�@7
%assignvariableop_65_vgg_net_kernel_16:@

identity_67��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Q
dtypesG
E2C		[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_variable_33Identity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_variable_32Identity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_variable_31Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_variable_30Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_variable_29Identity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_variable_28Identity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_variable_27Identity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpassignvariableop_7_variable_26Identity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpassignvariableop_8_variable_25Identity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOpassignvariableop_9_variable_24Identity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpassignvariableop_10_variable_23Identity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpassignvariableop_11_variable_22Identity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOpassignvariableop_12_variable_21Identity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOpassignvariableop_13_variable_20Identity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpassignvariableop_14_variable_19Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOpassignvariableop_15_variable_18Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpassignvariableop_16_variable_17Identity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOpassignvariableop_17_variable_16Identity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOpassignvariableop_18_variable_15Identity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_variable_14Identity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOpassignvariableop_20_variable_13Identity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_variable_12Identity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_11Identity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_variable_10Identity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_variable_9Identity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_variable_8Identity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_variable_7Identity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOpassignvariableop_27_variable_6Identity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_28AssignVariableOpassignvariableop_28_variable_5Identity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOpassignvariableop_29_variable_4Identity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_variable_3Identity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_31AssignVariableOpassignvariableop_31_variable_2Identity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_variable_1Identity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_variableIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp#assignvariableop_34_vgg_net_bias_28Identity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp%assignvariableop_35_vgg_net_kernel_27Identity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp#assignvariableop_36_vgg_net_bias_21Identity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp%assignvariableop_37_vgg_net_kernel_20Identity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp%assignvariableop_38_vgg_net_kernel_31Identity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp#assignvariableop_39_vgg_net_bias_27Identity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp%assignvariableop_40_vgg_net_kernel_26Identity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp%assignvariableop_41_vgg_net_kernel_23Identity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp#assignvariableop_42_vgg_net_bias_20Identity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp#assignvariableop_43_vgg_net_bias_19Identity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp#assignvariableop_44_vgg_net_bias_18Identity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp#assignvariableop_45_vgg_net_bias_17Identity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp#assignvariableop_46_vgg_net_bias_16Identity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp#assignvariableop_47_vgg_net_bias_31Identity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp%assignvariableop_48_vgg_net_kernel_30Identity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp%assignvariableop_49_vgg_net_kernel_25Identity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp#assignvariableop_50_vgg_net_bias_26Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp%assignvariableop_51_vgg_net_kernel_19Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp#assignvariableop_52_vgg_net_bias_23Identity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp%assignvariableop_53_vgg_net_kernel_22Identity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp#assignvariableop_54_vgg_net_bias_30Identity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp#assignvariableop_55_vgg_net_bias_29Identity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp%assignvariableop_56_vgg_net_kernel_28Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp#assignvariableop_57_vgg_net_bias_24Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp#assignvariableop_58_vgg_net_bias_25Identity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp%assignvariableop_59_vgg_net_kernel_29Identity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp%assignvariableop_60_vgg_net_kernel_24Identity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp#assignvariableop_61_vgg_net_bias_22Identity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp%assignvariableop_62_vgg_net_kernel_21Identity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp%assignvariableop_63_vgg_net_kernel_18Identity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp%assignvariableop_64_vgg_net_kernel_17Identity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp%assignvariableop_65_vgg_net_kernel_16Identity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_66Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_67IdentityIdentity_66:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_67Identity_67:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:1B-
+
_user_specified_namevgg_net/kernel_16:1A-
+
_user_specified_namevgg_net/kernel_17:1@-
+
_user_specified_namevgg_net/kernel_18:1?-
+
_user_specified_namevgg_net/kernel_21:/>+
)
_user_specified_namevgg_net/bias_22:1=-
+
_user_specified_namevgg_net/kernel_24:1<-
+
_user_specified_namevgg_net/kernel_29:/;+
)
_user_specified_namevgg_net/bias_25:/:+
)
_user_specified_namevgg_net/bias_24:19-
+
_user_specified_namevgg_net/kernel_28:/8+
)
_user_specified_namevgg_net/bias_29:/7+
)
_user_specified_namevgg_net/bias_30:16-
+
_user_specified_namevgg_net/kernel_22:/5+
)
_user_specified_namevgg_net/bias_23:14-
+
_user_specified_namevgg_net/kernel_19:/3+
)
_user_specified_namevgg_net/bias_26:12-
+
_user_specified_namevgg_net/kernel_25:11-
+
_user_specified_namevgg_net/kernel_30:/0+
)
_user_specified_namevgg_net/bias_31://+
)
_user_specified_namevgg_net/bias_16:/.+
)
_user_specified_namevgg_net/bias_17:/-+
)
_user_specified_namevgg_net/bias_18:/,+
)
_user_specified_namevgg_net/bias_19:/++
)
_user_specified_namevgg_net/bias_20:1*-
+
_user_specified_namevgg_net/kernel_23:1)-
+
_user_specified_namevgg_net/kernel_26:/(+
)
_user_specified_namevgg_net/bias_27:1'-
+
_user_specified_namevgg_net/kernel_31:1&-
+
_user_specified_namevgg_net/kernel_20:/%+
)
_user_specified_namevgg_net/bias_21:1$-
+
_user_specified_namevgg_net/kernel_27:/#+
)
_user_specified_namevgg_net/bias_28:("$
"
_user_specified_name
Variable:*!&
$
_user_specified_name
Variable_1:* &
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+
'
%
_user_specified_nameVariable_24:+	'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_28:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_31:+'
%
_user_specified_nameVariable_32:+'
%
_user_specified_nameVariable_33:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
-__inference_signature_wrapper___call___129905

args_0!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�@

unknown_28:@

unknown_29:@


unknown_30:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___129766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_name129901:&"
 
_user_specified_name129899:&"
 
_user_specified_name129897:&"
 
_user_specified_name129895:&"
 
_user_specified_name129893:&"
 
_user_specified_name129891:&"
 
_user_specified_name129889:&"
 
_user_specified_name129887:&"
 
_user_specified_name129885:&"
 
_user_specified_name129883:&"
 
_user_specified_name129881:&"
 
_user_specified_name129879:&"
 
_user_specified_name129877:&"
 
_user_specified_name129875:&"
 
_user_specified_name129873:&"
 
_user_specified_name129871:&"
 
_user_specified_name129869:&"
 
_user_specified_name129867:&"
 
_user_specified_name129865:&"
 
_user_specified_name129863:&"
 
_user_specified_name129861:&"
 
_user_specified_name129859:&
"
 
_user_specified_name129857:&	"
 
_user_specified_name129855:&"
 
_user_specified_name129853:&"
 
_user_specified_name129851:&"
 
_user_specified_name129849:&"
 
_user_specified_name129847:&"
 
_user_specified_name129845:&"
 
_user_specified_name129843:&"
 
_user_specified_name129841:&"
 
_user_specified_name129839:W S
/
_output_shapes
:���������@@
 
_user_specified_nameargs_0
�
�
-__inference_signature_wrapper___call___129836

args_0!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7: @
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@%

unknown_13:@�

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:
��

unknown_26:	�

unknown_27:	�@

unknown_28:@

unknown_29:@


unknown_30:

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallargs_0unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������
*B
_read_only_resource_inputs$
" 	
 *2
config_proto" 

CPU

GPU 2J 8� �J *$
fR
__inference___call___129766o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������
<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_name129832:&"
 
_user_specified_name129830:&"
 
_user_specified_name129828:&"
 
_user_specified_name129826:&"
 
_user_specified_name129824:&"
 
_user_specified_name129822:&"
 
_user_specified_name129820:&"
 
_user_specified_name129818:&"
 
_user_specified_name129816:&"
 
_user_specified_name129814:&"
 
_user_specified_name129812:&"
 
_user_specified_name129810:&"
 
_user_specified_name129808:&"
 
_user_specified_name129806:&"
 
_user_specified_name129804:&"
 
_user_specified_name129802:&"
 
_user_specified_name129800:&"
 
_user_specified_name129798:&"
 
_user_specified_name129796:&"
 
_user_specified_name129794:&"
 
_user_specified_name129792:&"
 
_user_specified_name129790:&
"
 
_user_specified_name129788:&	"
 
_user_specified_name129786:&"
 
_user_specified_name129784:&"
 
_user_specified_name129782:&"
 
_user_specified_name129780:&"
 
_user_specified_name129778:&"
 
_user_specified_name129776:&"
 
_user_specified_name129774:&"
 
_user_specified_name129772:&"
 
_user_specified_name129770:W S
/
_output_shapes
:���������@@
 
_user_specified_nameargs_0
��
�
__inference___call___129766

args_0P
6vgg_net_1_conv2d_1_convolution_readvariableop_resource:@
2vgg_net_1_conv2d_1_reshape_readvariableop_resource:R
8vgg_net_1_conv2d_1_2_convolution_readvariableop_resource:B
4vgg_net_1_conv2d_1_2_reshape_readvariableop_resource:R
8vgg_net_1_conv2d_2_1_convolution_readvariableop_resource: B
4vgg_net_1_conv2d_2_1_reshape_readvariableop_resource: R
8vgg_net_1_conv2d_3_1_convolution_readvariableop_resource:  B
4vgg_net_1_conv2d_3_1_reshape_readvariableop_resource: R
8vgg_net_1_conv2d_4_1_convolution_readvariableop_resource: @B
4vgg_net_1_conv2d_4_1_reshape_readvariableop_resource:@R
8vgg_net_1_conv2d_5_1_convolution_readvariableop_resource:@@B
4vgg_net_1_conv2d_5_1_reshape_readvariableop_resource:@R
8vgg_net_1_conv2d_6_1_convolution_readvariableop_resource:@@B
4vgg_net_1_conv2d_6_1_reshape_readvariableop_resource:@S
8vgg_net_1_conv2d_7_1_convolution_readvariableop_resource:@�C
4vgg_net_1_conv2d_7_1_reshape_readvariableop_resource:	�T
8vgg_net_1_conv2d_8_1_convolution_readvariableop_resource:��C
4vgg_net_1_conv2d_8_1_reshape_readvariableop_resource:	�T
8vgg_net_1_conv2d_9_1_convolution_readvariableop_resource:��C
4vgg_net_1_conv2d_9_1_reshape_readvariableop_resource:	�U
9vgg_net_1_conv2d_10_1_convolution_readvariableop_resource:��D
5vgg_net_1_conv2d_10_1_reshape_readvariableop_resource:	�U
9vgg_net_1_conv2d_11_1_convolution_readvariableop_resource:��D
5vgg_net_1_conv2d_11_1_reshape_readvariableop_resource:	�U
9vgg_net_1_conv2d_12_1_convolution_readvariableop_resource:��D
5vgg_net_1_conv2d_12_1_reshape_readvariableop_resource:	�B
.vgg_net_1_dense_1_cast_readvariableop_resource:
��@
1vgg_net_1_dense_1_biasadd_readvariableop_resource:	�C
0vgg_net_1_dense_1_2_cast_readvariableop_resource:	�@A
3vgg_net_1_dense_1_2_biasadd_readvariableop_resource:@B
0vgg_net_1_dense_2_1_cast_readvariableop_resource:@
A
3vgg_net_1_dense_2_1_biasadd_readvariableop_resource:

identity��)vgg_net_1/conv2d_1/Reshape/ReadVariableOp�-vgg_net_1/conv2d_1/convolution/ReadVariableOp�,vgg_net_1/conv2d_10_1/Reshape/ReadVariableOp�0vgg_net_1/conv2d_10_1/convolution/ReadVariableOp�,vgg_net_1/conv2d_11_1/Reshape/ReadVariableOp�0vgg_net_1/conv2d_11_1/convolution/ReadVariableOp�,vgg_net_1/conv2d_12_1/Reshape/ReadVariableOp�0vgg_net_1/conv2d_12_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_1_2/Reshape/ReadVariableOp�/vgg_net_1/conv2d_1_2/convolution/ReadVariableOp�+vgg_net_1/conv2d_2_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_2_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_3_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_3_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_4_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_4_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_5_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_5_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_6_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_6_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_7_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_7_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_8_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_8_1/convolution/ReadVariableOp�+vgg_net_1/conv2d_9_1/Reshape/ReadVariableOp�/vgg_net_1/conv2d_9_1/convolution/ReadVariableOp�(vgg_net_1/dense_1/BiasAdd/ReadVariableOp�%vgg_net_1/dense_1/Cast/ReadVariableOp�*vgg_net_1/dense_1_2/BiasAdd/ReadVariableOp�'vgg_net_1/dense_1_2/Cast/ReadVariableOp�*vgg_net_1/dense_2_1/BiasAdd/ReadVariableOp�'vgg_net_1/dense_2_1/Cast/ReadVariableOp�
-vgg_net_1/conv2d_1/convolution/ReadVariableOpReadVariableOp6vgg_net_1_conv2d_1_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
vgg_net_1/conv2d_1/convolutionConv2Dargs_05vgg_net_1/conv2d_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
)vgg_net_1/conv2d_1/Reshape/ReadVariableOpReadVariableOp2vgg_net_1_conv2d_1_reshape_readvariableop_resource*
_output_shapes
:*
dtype0y
 vgg_net_1/conv2d_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
vgg_net_1/conv2d_1/ReshapeReshape1vgg_net_1/conv2d_1/Reshape/ReadVariableOp:value:0)vgg_net_1/conv2d_1/Reshape/shape:output:0*
T0*&
_output_shapes
:o
vgg_net_1/conv2d_1/SqueezeSqueeze#vgg_net_1/conv2d_1/Reshape:output:0*
T0*
_output_shapes
:�
vgg_net_1/conv2d_1/BiasAddBiasAdd'vgg_net_1/conv2d_1/convolution:output:0#vgg_net_1/conv2d_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@@~
vgg_net_1/conv2d_1/ReluRelu#vgg_net_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@�
/vgg_net_1/conv2d_1_2/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_1_2_convolution_readvariableop_resource*&
_output_shapes
:*
dtype0�
 vgg_net_1/conv2d_1_2/convolutionConv2D%vgg_net_1/conv2d_1/Relu:activations:07vgg_net_1/conv2d_1_2/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@*
paddingSAME*
strides
�
+vgg_net_1/conv2d_1_2/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_1_2_reshape_readvariableop_resource*
_output_shapes
:*
dtype0{
"vgg_net_1/conv2d_1_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            �
vgg_net_1/conv2d_1_2/ReshapeReshape3vgg_net_1/conv2d_1_2/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_1_2/Reshape/shape:output:0*
T0*&
_output_shapes
:s
vgg_net_1/conv2d_1_2/SqueezeSqueeze%vgg_net_1/conv2d_1_2/Reshape:output:0*
T0*
_output_shapes
:�
vgg_net_1/conv2d_1_2/BiasAddBiasAdd)vgg_net_1/conv2d_1_2/convolution:output:0%vgg_net_1/conv2d_1_2/Squeeze:output:0*
T0*/
_output_shapes
:���������@@�
vgg_net_1/conv2d_1_2/ReluRelu%vgg_net_1/conv2d_1_2/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@�
#vgg_net_1/max_pooling2d_1/MaxPool2dMaxPool'vgg_net_1/conv2d_1_2/Relu:activations:0*/
_output_shapes
:���������  *
ksize
*
paddingVALID*
strides
�
/vgg_net_1/conv2d_2_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_2_1_convolution_readvariableop_resource*&
_output_shapes
: *
dtype0�
 vgg_net_1/conv2d_2_1/convolutionConv2D,vgg_net_1/max_pooling2d_1/MaxPool2d:output:07vgg_net_1/conv2d_2_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
+vgg_net_1/conv2d_2_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_2_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0{
"vgg_net_1/conv2d_2_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
vgg_net_1/conv2d_2_1/ReshapeReshape3vgg_net_1/conv2d_2_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_2_1/Reshape/shape:output:0*
T0*&
_output_shapes
: s
vgg_net_1/conv2d_2_1/SqueezeSqueeze%vgg_net_1/conv2d_2_1/Reshape:output:0*
T0*
_output_shapes
: �
vgg_net_1/conv2d_2_1/BiasAddBiasAdd)vgg_net_1/conv2d_2_1/convolution:output:0%vgg_net_1/conv2d_2_1/Squeeze:output:0*
T0*/
_output_shapes
:���������   �
vgg_net_1/conv2d_2_1/ReluRelu%vgg_net_1/conv2d_2_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
/vgg_net_1/conv2d_3_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_3_1_convolution_readvariableop_resource*&
_output_shapes
:  *
dtype0�
 vgg_net_1/conv2d_3_1/convolutionConv2D'vgg_net_1/conv2d_2_1/Relu:activations:07vgg_net_1/conv2d_3_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������   *
paddingSAME*
strides
�
+vgg_net_1/conv2d_3_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_3_1_reshape_readvariableop_resource*
_output_shapes
: *
dtype0{
"vgg_net_1/conv2d_3_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             �
vgg_net_1/conv2d_3_1/ReshapeReshape3vgg_net_1/conv2d_3_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_3_1/Reshape/shape:output:0*
T0*&
_output_shapes
: s
vgg_net_1/conv2d_3_1/SqueezeSqueeze%vgg_net_1/conv2d_3_1/Reshape:output:0*
T0*
_output_shapes
: �
vgg_net_1/conv2d_3_1/BiasAddBiasAdd)vgg_net_1/conv2d_3_1/convolution:output:0%vgg_net_1/conv2d_3_1/Squeeze:output:0*
T0*/
_output_shapes
:���������   �
vgg_net_1/conv2d_3_1/ReluRelu%vgg_net_1/conv2d_3_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������   �
%vgg_net_1/max_pooling2d_1_2/MaxPool2dMaxPool'vgg_net_1/conv2d_3_1/Relu:activations:0*/
_output_shapes
:��������� *
ksize
*
paddingVALID*
strides
�
/vgg_net_1/conv2d_4_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_4_1_convolution_readvariableop_resource*&
_output_shapes
: @*
dtype0�
 vgg_net_1/conv2d_4_1/convolutionConv2D.vgg_net_1/max_pooling2d_1_2/MaxPool2d:output:07vgg_net_1/conv2d_4_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
+vgg_net_1/conv2d_4_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_4_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0{
"vgg_net_1/conv2d_4_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
vgg_net_1/conv2d_4_1/ReshapeReshape3vgg_net_1/conv2d_4_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_4_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@s
vgg_net_1/conv2d_4_1/SqueezeSqueeze%vgg_net_1/conv2d_4_1/Reshape:output:0*
T0*
_output_shapes
:@�
vgg_net_1/conv2d_4_1/BiasAddBiasAdd)vgg_net_1/conv2d_4_1/convolution:output:0%vgg_net_1/conv2d_4_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@�
vgg_net_1/conv2d_4_1/ReluRelu%vgg_net_1/conv2d_4_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
/vgg_net_1/conv2d_5_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_5_1_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
 vgg_net_1/conv2d_5_1/convolutionConv2D'vgg_net_1/conv2d_4_1/Relu:activations:07vgg_net_1/conv2d_5_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
+vgg_net_1/conv2d_5_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_5_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0{
"vgg_net_1/conv2d_5_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
vgg_net_1/conv2d_5_1/ReshapeReshape3vgg_net_1/conv2d_5_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_5_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@s
vgg_net_1/conv2d_5_1/SqueezeSqueeze%vgg_net_1/conv2d_5_1/Reshape:output:0*
T0*
_output_shapes
:@�
vgg_net_1/conv2d_5_1/BiasAddBiasAdd)vgg_net_1/conv2d_5_1/convolution:output:0%vgg_net_1/conv2d_5_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@�
vgg_net_1/conv2d_5_1/ReluRelu%vgg_net_1/conv2d_5_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
/vgg_net_1/conv2d_6_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_6_1_convolution_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
 vgg_net_1/conv2d_6_1/convolutionConv2D'vgg_net_1/conv2d_5_1/Relu:activations:07vgg_net_1/conv2d_6_1/convolution/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
�
+vgg_net_1/conv2d_6_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_6_1_reshape_readvariableop_resource*
_output_shapes
:@*
dtype0{
"vgg_net_1/conv2d_6_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         @   �
vgg_net_1/conv2d_6_1/ReshapeReshape3vgg_net_1/conv2d_6_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_6_1/Reshape/shape:output:0*
T0*&
_output_shapes
:@s
vgg_net_1/conv2d_6_1/SqueezeSqueeze%vgg_net_1/conv2d_6_1/Reshape:output:0*
T0*
_output_shapes
:@�
vgg_net_1/conv2d_6_1/BiasAddBiasAdd)vgg_net_1/conv2d_6_1/convolution:output:0%vgg_net_1/conv2d_6_1/Squeeze:output:0*
T0*/
_output_shapes
:���������@�
vgg_net_1/conv2d_6_1/ReluRelu%vgg_net_1/conv2d_6_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@�
%vgg_net_1/max_pooling2d_2_1/MaxPool2dMaxPool'vgg_net_1/conv2d_6_1/Relu:activations:0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
�
/vgg_net_1/conv2d_7_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_7_1_convolution_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
 vgg_net_1/conv2d_7_1/convolutionConv2D.vgg_net_1/max_pooling2d_2_1/MaxPool2d:output:07vgg_net_1/conv2d_7_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
+vgg_net_1/conv2d_7_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_7_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0{
"vgg_net_1/conv2d_7_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
vgg_net_1/conv2d_7_1/ReshapeReshape3vgg_net_1/conv2d_7_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_7_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�t
vgg_net_1/conv2d_7_1/SqueezeSqueeze%vgg_net_1/conv2d_7_1/Reshape:output:0*
T0*
_output_shapes	
:��
vgg_net_1/conv2d_7_1/BiasAddBiasAdd)vgg_net_1/conv2d_7_1/convolution:output:0%vgg_net_1/conv2d_7_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
vgg_net_1/conv2d_7_1/ReluRelu%vgg_net_1/conv2d_7_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
/vgg_net_1/conv2d_8_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_8_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 vgg_net_1/conv2d_8_1/convolutionConv2D'vgg_net_1/conv2d_7_1/Relu:activations:07vgg_net_1/conv2d_8_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
+vgg_net_1/conv2d_8_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_8_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0{
"vgg_net_1/conv2d_8_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
vgg_net_1/conv2d_8_1/ReshapeReshape3vgg_net_1/conv2d_8_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_8_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�t
vgg_net_1/conv2d_8_1/SqueezeSqueeze%vgg_net_1/conv2d_8_1/Reshape:output:0*
T0*
_output_shapes	
:��
vgg_net_1/conv2d_8_1/BiasAddBiasAdd)vgg_net_1/conv2d_8_1/convolution:output:0%vgg_net_1/conv2d_8_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
vgg_net_1/conv2d_8_1/ReluRelu%vgg_net_1/conv2d_8_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
/vgg_net_1/conv2d_9_1/convolution/ReadVariableOpReadVariableOp8vgg_net_1_conv2d_9_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
 vgg_net_1/conv2d_9_1/convolutionConv2D'vgg_net_1/conv2d_8_1/Relu:activations:07vgg_net_1/conv2d_9_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
+vgg_net_1/conv2d_9_1/Reshape/ReadVariableOpReadVariableOp4vgg_net_1_conv2d_9_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0{
"vgg_net_1/conv2d_9_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
vgg_net_1/conv2d_9_1/ReshapeReshape3vgg_net_1/conv2d_9_1/Reshape/ReadVariableOp:value:0+vgg_net_1/conv2d_9_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�t
vgg_net_1/conv2d_9_1/SqueezeSqueeze%vgg_net_1/conv2d_9_1/Reshape:output:0*
T0*
_output_shapes	
:��
vgg_net_1/conv2d_9_1/BiasAddBiasAdd)vgg_net_1/conv2d_9_1/convolution:output:0%vgg_net_1/conv2d_9_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
vgg_net_1/conv2d_9_1/ReluRelu%vgg_net_1/conv2d_9_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
%vgg_net_1/max_pooling2d_3_1/MaxPool2dMaxPool'vgg_net_1/conv2d_9_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
0vgg_net_1/conv2d_10_1/convolution/ReadVariableOpReadVariableOp9vgg_net_1_conv2d_10_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
!vgg_net_1/conv2d_10_1/convolutionConv2D.vgg_net_1/max_pooling2d_3_1/MaxPool2d:output:08vgg_net_1/conv2d_10_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
,vgg_net_1/conv2d_10_1/Reshape/ReadVariableOpReadVariableOp5vgg_net_1_conv2d_10_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0|
#vgg_net_1/conv2d_10_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
vgg_net_1/conv2d_10_1/ReshapeReshape4vgg_net_1/conv2d_10_1/Reshape/ReadVariableOp:value:0,vgg_net_1/conv2d_10_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�v
vgg_net_1/conv2d_10_1/SqueezeSqueeze&vgg_net_1/conv2d_10_1/Reshape:output:0*
T0*
_output_shapes	
:��
vgg_net_1/conv2d_10_1/BiasAddBiasAdd*vgg_net_1/conv2d_10_1/convolution:output:0&vgg_net_1/conv2d_10_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
vgg_net_1/conv2d_10_1/ReluRelu&vgg_net_1/conv2d_10_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
0vgg_net_1/conv2d_11_1/convolution/ReadVariableOpReadVariableOp9vgg_net_1_conv2d_11_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
!vgg_net_1/conv2d_11_1/convolutionConv2D(vgg_net_1/conv2d_10_1/Relu:activations:08vgg_net_1/conv2d_11_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
,vgg_net_1/conv2d_11_1/Reshape/ReadVariableOpReadVariableOp5vgg_net_1_conv2d_11_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0|
#vgg_net_1/conv2d_11_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
vgg_net_1/conv2d_11_1/ReshapeReshape4vgg_net_1/conv2d_11_1/Reshape/ReadVariableOp:value:0,vgg_net_1/conv2d_11_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�v
vgg_net_1/conv2d_11_1/SqueezeSqueeze&vgg_net_1/conv2d_11_1/Reshape:output:0*
T0*
_output_shapes	
:��
vgg_net_1/conv2d_11_1/BiasAddBiasAdd*vgg_net_1/conv2d_11_1/convolution:output:0&vgg_net_1/conv2d_11_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
vgg_net_1/conv2d_11_1/ReluRelu&vgg_net_1/conv2d_11_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
0vgg_net_1/conv2d_12_1/convolution/ReadVariableOpReadVariableOp9vgg_net_1_conv2d_12_1_convolution_readvariableop_resource*(
_output_shapes
:��*
dtype0�
!vgg_net_1/conv2d_12_1/convolutionConv2D(vgg_net_1/conv2d_11_1/Relu:activations:08vgg_net_1/conv2d_12_1/convolution/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
,vgg_net_1/conv2d_12_1/Reshape/ReadVariableOpReadVariableOp5vgg_net_1_conv2d_12_1_reshape_readvariableop_resource*
_output_shapes	
:�*
dtype0|
#vgg_net_1/conv2d_12_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         �   �
vgg_net_1/conv2d_12_1/ReshapeReshape4vgg_net_1/conv2d_12_1/Reshape/ReadVariableOp:value:0,vgg_net_1/conv2d_12_1/Reshape/shape:output:0*
T0*'
_output_shapes
:�v
vgg_net_1/conv2d_12_1/SqueezeSqueeze&vgg_net_1/conv2d_12_1/Reshape:output:0*
T0*
_output_shapes	
:��
vgg_net_1/conv2d_12_1/BiasAddBiasAdd*vgg_net_1/conv2d_12_1/convolution:output:0&vgg_net_1/conv2d_12_1/Squeeze:output:0*
T0*0
_output_shapes
:�����������
vgg_net_1/conv2d_12_1/ReluRelu&vgg_net_1/conv2d_12_1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
%vgg_net_1/max_pooling2d_4_1/MaxPool2dMaxPool(vgg_net_1/conv2d_12_1/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
r
!vgg_net_1/flatten_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"����   �
vgg_net_1/flatten_1/ReshapeReshape.vgg_net_1/max_pooling2d_4_1/MaxPool2d:output:0*vgg_net_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:�����������
%vgg_net_1/dense_1/Cast/ReadVariableOpReadVariableOp.vgg_net_1_dense_1_cast_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
vgg_net_1/dense_1/MatMulMatMul$vgg_net_1/flatten_1/Reshape:output:0-vgg_net_1/dense_1/Cast/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(vgg_net_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp1vgg_net_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
vgg_net_1/dense_1/BiasAddBiasAdd"vgg_net_1/dense_1/MatMul:product:00vgg_net_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
vgg_net_1/dense_1/ReluRelu"vgg_net_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'vgg_net_1/dense_1_2/Cast/ReadVariableOpReadVariableOp0vgg_net_1_dense_1_2_cast_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
vgg_net_1/dense_1_2/MatMulMatMul$vgg_net_1/dense_1/Relu:activations:0/vgg_net_1/dense_1_2/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
*vgg_net_1/dense_1_2/BiasAdd/ReadVariableOpReadVariableOp3vgg_net_1_dense_1_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
vgg_net_1/dense_1_2/BiasAddBiasAdd$vgg_net_1/dense_1_2/MatMul:product:02vgg_net_1/dense_1_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
vgg_net_1/dense_1_2/ReluRelu$vgg_net_1/dense_1_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
'vgg_net_1/dense_2_1/Cast/ReadVariableOpReadVariableOp0vgg_net_1_dense_2_1_cast_readvariableop_resource*
_output_shapes

:@
*
dtype0�
vgg_net_1/dense_2_1/MatMulMatMul&vgg_net_1/dense_1_2/Relu:activations:0/vgg_net_1/dense_2_1/Cast/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
�
*vgg_net_1/dense_2_1/BiasAdd/ReadVariableOpReadVariableOp3vgg_net_1_dense_2_1_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0�
vgg_net_1/dense_2_1/BiasAddBiasAdd$vgg_net_1/dense_2_1/MatMul:product:02vgg_net_1/dense_2_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������
~
vgg_net_1/dense_2_1/SoftmaxSoftmax$vgg_net_1/dense_2_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������
t
IdentityIdentity%vgg_net_1/dense_2_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������
�
NoOpNoOp*^vgg_net_1/conv2d_1/Reshape/ReadVariableOp.^vgg_net_1/conv2d_1/convolution/ReadVariableOp-^vgg_net_1/conv2d_10_1/Reshape/ReadVariableOp1^vgg_net_1/conv2d_10_1/convolution/ReadVariableOp-^vgg_net_1/conv2d_11_1/Reshape/ReadVariableOp1^vgg_net_1/conv2d_11_1/convolution/ReadVariableOp-^vgg_net_1/conv2d_12_1/Reshape/ReadVariableOp1^vgg_net_1/conv2d_12_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_1_2/Reshape/ReadVariableOp0^vgg_net_1/conv2d_1_2/convolution/ReadVariableOp,^vgg_net_1/conv2d_2_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_2_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_3_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_3_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_4_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_4_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_5_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_5_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_6_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_6_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_7_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_7_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_8_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_8_1/convolution/ReadVariableOp,^vgg_net_1/conv2d_9_1/Reshape/ReadVariableOp0^vgg_net_1/conv2d_9_1/convolution/ReadVariableOp)^vgg_net_1/dense_1/BiasAdd/ReadVariableOp&^vgg_net_1/dense_1/Cast/ReadVariableOp+^vgg_net_1/dense_1_2/BiasAdd/ReadVariableOp(^vgg_net_1/dense_1_2/Cast/ReadVariableOp+^vgg_net_1/dense_2_1/BiasAdd/ReadVariableOp(^vgg_net_1/dense_2_1/Cast/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*n
_input_shapes]
[:���������@@: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2V
)vgg_net_1/conv2d_1/Reshape/ReadVariableOp)vgg_net_1/conv2d_1/Reshape/ReadVariableOp2^
-vgg_net_1/conv2d_1/convolution/ReadVariableOp-vgg_net_1/conv2d_1/convolution/ReadVariableOp2\
,vgg_net_1/conv2d_10_1/Reshape/ReadVariableOp,vgg_net_1/conv2d_10_1/Reshape/ReadVariableOp2d
0vgg_net_1/conv2d_10_1/convolution/ReadVariableOp0vgg_net_1/conv2d_10_1/convolution/ReadVariableOp2\
,vgg_net_1/conv2d_11_1/Reshape/ReadVariableOp,vgg_net_1/conv2d_11_1/Reshape/ReadVariableOp2d
0vgg_net_1/conv2d_11_1/convolution/ReadVariableOp0vgg_net_1/conv2d_11_1/convolution/ReadVariableOp2\
,vgg_net_1/conv2d_12_1/Reshape/ReadVariableOp,vgg_net_1/conv2d_12_1/Reshape/ReadVariableOp2d
0vgg_net_1/conv2d_12_1/convolution/ReadVariableOp0vgg_net_1/conv2d_12_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_1_2/Reshape/ReadVariableOp+vgg_net_1/conv2d_1_2/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_1_2/convolution/ReadVariableOp/vgg_net_1/conv2d_1_2/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_2_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_2_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_2_1/convolution/ReadVariableOp/vgg_net_1/conv2d_2_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_3_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_3_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_3_1/convolution/ReadVariableOp/vgg_net_1/conv2d_3_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_4_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_4_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_4_1/convolution/ReadVariableOp/vgg_net_1/conv2d_4_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_5_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_5_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_5_1/convolution/ReadVariableOp/vgg_net_1/conv2d_5_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_6_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_6_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_6_1/convolution/ReadVariableOp/vgg_net_1/conv2d_6_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_7_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_7_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_7_1/convolution/ReadVariableOp/vgg_net_1/conv2d_7_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_8_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_8_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_8_1/convolution/ReadVariableOp/vgg_net_1/conv2d_8_1/convolution/ReadVariableOp2Z
+vgg_net_1/conv2d_9_1/Reshape/ReadVariableOp+vgg_net_1/conv2d_9_1/Reshape/ReadVariableOp2b
/vgg_net_1/conv2d_9_1/convolution/ReadVariableOp/vgg_net_1/conv2d_9_1/convolution/ReadVariableOp2T
(vgg_net_1/dense_1/BiasAdd/ReadVariableOp(vgg_net_1/dense_1/BiasAdd/ReadVariableOp2N
%vgg_net_1/dense_1/Cast/ReadVariableOp%vgg_net_1/dense_1/Cast/ReadVariableOp2X
*vgg_net_1/dense_1_2/BiasAdd/ReadVariableOp*vgg_net_1/dense_1_2/BiasAdd/ReadVariableOp2R
'vgg_net_1/dense_1_2/Cast/ReadVariableOp'vgg_net_1/dense_1_2/Cast/ReadVariableOp2X
*vgg_net_1/dense_2_1/BiasAdd/ReadVariableOp*vgg_net_1/dense_2_1/BiasAdd/ReadVariableOp2R
'vgg_net_1/dense_2_1/Cast/ReadVariableOp'vgg_net_1/dense_2_1/Cast/ReadVariableOp:( $
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameargs_0
��
�;
__inference__traced_save_130461
file_prefix<
"read_disablecopyonread_variable_33:2
$read_1_disablecopyonread_variable_32:>
$read_2_disablecopyonread_variable_31:2
$read_3_disablecopyonread_variable_30:>
$read_4_disablecopyonread_variable_29: 2
$read_5_disablecopyonread_variable_28: >
$read_6_disablecopyonread_variable_27:  2
$read_7_disablecopyonread_variable_26: >
$read_8_disablecopyonread_variable_25: @2
$read_9_disablecopyonread_variable_24:@?
%read_10_disablecopyonread_variable_23:@@3
%read_11_disablecopyonread_variable_22:@?
%read_12_disablecopyonread_variable_21:@@3
%read_13_disablecopyonread_variable_20:@@
%read_14_disablecopyonread_variable_19:@�4
%read_15_disablecopyonread_variable_18:	�A
%read_16_disablecopyonread_variable_17:��4
%read_17_disablecopyonread_variable_16:	�A
%read_18_disablecopyonread_variable_15:��4
%read_19_disablecopyonread_variable_14:	�A
%read_20_disablecopyonread_variable_13:��4
%read_21_disablecopyonread_variable_12:	�A
%read_22_disablecopyonread_variable_11:��4
%read_23_disablecopyonread_variable_10:	�@
$read_24_disablecopyonread_variable_9:��3
$read_25_disablecopyonread_variable_8:	�8
$read_26_disablecopyonread_variable_7:
��3
$read_27_disablecopyonread_variable_6:	�2
$read_28_disablecopyonread_variable_5:	7
$read_29_disablecopyonread_variable_4:	�@2
$read_30_disablecopyonread_variable_3:@2
$read_31_disablecopyonread_variable_2:	6
$read_32_disablecopyonread_variable_1:@
0
"read_33_disablecopyonread_variable:
7
)read_34_disablecopyonread_vgg_net_bias_28: E
+read_35_disablecopyonread_vgg_net_kernel_27: @8
)read_36_disablecopyonread_vgg_net_bias_21:	�G
+read_37_disablecopyonread_vgg_net_kernel_20:��E
+read_38_disablecopyonread_vgg_net_kernel_31:7
)read_39_disablecopyonread_vgg_net_bias_27:@E
+read_40_disablecopyonread_vgg_net_kernel_26:@@G
+read_41_disablecopyonread_vgg_net_kernel_23:��8
)read_42_disablecopyonread_vgg_net_bias_20:	�8
)read_43_disablecopyonread_vgg_net_bias_19:	�8
)read_44_disablecopyonread_vgg_net_bias_18:	�7
)read_45_disablecopyonread_vgg_net_bias_17:@7
)read_46_disablecopyonread_vgg_net_bias_16:
7
)read_47_disablecopyonread_vgg_net_bias_31:E
+read_48_disablecopyonread_vgg_net_kernel_30:E
+read_49_disablecopyonread_vgg_net_kernel_25:@@7
)read_50_disablecopyonread_vgg_net_bias_26:@G
+read_51_disablecopyonread_vgg_net_kernel_19:��8
)read_52_disablecopyonread_vgg_net_bias_23:	�G
+read_53_disablecopyonread_vgg_net_kernel_22:��7
)read_54_disablecopyonread_vgg_net_bias_30:7
)read_55_disablecopyonread_vgg_net_bias_29: E
+read_56_disablecopyonread_vgg_net_kernel_28:  8
)read_57_disablecopyonread_vgg_net_bias_24:	�7
)read_58_disablecopyonread_vgg_net_bias_25:@E
+read_59_disablecopyonread_vgg_net_kernel_29: F
+read_60_disablecopyonread_vgg_net_kernel_24:@�8
)read_61_disablecopyonread_vgg_net_bias_22:	�G
+read_62_disablecopyonread_vgg_net_kernel_21:��?
+read_63_disablecopyonread_vgg_net_kernel_18:
��>
+read_64_disablecopyonread_vgg_net_kernel_17:	�@=
+read_65_disablecopyonread_vgg_net_kernel_16:@

savev2_const
identity_133��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
: e
Read/DisableCopyOnReadDisableCopyOnRead"read_disablecopyonread_variable_33*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp"read_disablecopyonread_variable_33^Read/DisableCopyOnRead*&
_output_shapes
:*
dtype0b
IdentityIdentityRead/ReadVariableOp:value:0*
T0*&
_output_shapes
:i

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:i
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_variable_32*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_variable_32^Read_1/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_2IdentityRead_1/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_2/DisableCopyOnReadDisableCopyOnRead$read_2_disablecopyonread_variable_31*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp$read_2_disablecopyonread_variable_31^Read_2/DisableCopyOnRead*&
_output_shapes
:*
dtype0f

Identity_4IdentityRead_2/ReadVariableOp:value:0*
T0*&
_output_shapes
:k

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*&
_output_shapes
:i
Read_3/DisableCopyOnReadDisableCopyOnRead$read_3_disablecopyonread_variable_30*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp$read_3_disablecopyonread_variable_30^Read_3/DisableCopyOnRead*
_output_shapes
:*
dtype0Z

Identity_6IdentityRead_3/ReadVariableOp:value:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:i
Read_4/DisableCopyOnReadDisableCopyOnRead$read_4_disablecopyonread_variable_29*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp$read_4_disablecopyonread_variable_29^Read_4/DisableCopyOnRead*&
_output_shapes
: *
dtype0f

Identity_8IdentityRead_4/ReadVariableOp:value:0*
T0*&
_output_shapes
: k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
: i
Read_5/DisableCopyOnReadDisableCopyOnRead$read_5_disablecopyonread_variable_28*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp$read_5_disablecopyonread_variable_28^Read_5/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_10IdentityRead_5/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_6/DisableCopyOnReadDisableCopyOnRead$read_6_disablecopyonread_variable_27*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp$read_6_disablecopyonread_variable_27^Read_6/DisableCopyOnRead*&
_output_shapes
:  *
dtype0g
Identity_12IdentityRead_6/ReadVariableOp:value:0*
T0*&
_output_shapes
:  m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:  i
Read_7/DisableCopyOnReadDisableCopyOnRead$read_7_disablecopyonread_variable_26*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp$read_7_disablecopyonread_variable_26^Read_7/DisableCopyOnRead*
_output_shapes
: *
dtype0[
Identity_14IdentityRead_7/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
: i
Read_8/DisableCopyOnReadDisableCopyOnRead$read_8_disablecopyonread_variable_25*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp$read_8_disablecopyonread_variable_25^Read_8/DisableCopyOnRead*&
_output_shapes
: @*
dtype0g
Identity_16IdentityRead_8/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*&
_output_shapes
: @i
Read_9/DisableCopyOnReadDisableCopyOnRead$read_9_disablecopyonread_variable_24*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp$read_9_disablecopyonread_variable_24^Read_9/DisableCopyOnRead*
_output_shapes
:@*
dtype0[
Identity_18IdentityRead_9/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_10/DisableCopyOnReadDisableCopyOnRead%read_10_disablecopyonread_variable_23*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp%read_10_disablecopyonread_variable_23^Read_10/DisableCopyOnRead*&
_output_shapes
:@@*
dtype0h
Identity_20IdentityRead_10/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@m
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@k
Read_11/DisableCopyOnReadDisableCopyOnRead%read_11_disablecopyonread_variable_22*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp%read_11_disablecopyonread_variable_22^Read_11/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_22IdentityRead_11/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_12/DisableCopyOnReadDisableCopyOnRead%read_12_disablecopyonread_variable_21*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp%read_12_disablecopyonread_variable_21^Read_12/DisableCopyOnRead*&
_output_shapes
:@@*
dtype0h
Identity_24IdentityRead_12/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@k
Read_13/DisableCopyOnReadDisableCopyOnRead%read_13_disablecopyonread_variable_20*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp%read_13_disablecopyonread_variable_20^Read_13/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_26IdentityRead_13/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@k
Read_14/DisableCopyOnReadDisableCopyOnRead%read_14_disablecopyonread_variable_19*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp%read_14_disablecopyonread_variable_19^Read_14/DisableCopyOnRead*'
_output_shapes
:@�*
dtype0i
Identity_28IdentityRead_14/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�n
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�k
Read_15/DisableCopyOnReadDisableCopyOnRead%read_15_disablecopyonread_variable_18*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp%read_15_disablecopyonread_variable_18^Read_15/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_30IdentityRead_15/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_16/DisableCopyOnReadDisableCopyOnRead%read_16_disablecopyonread_variable_17*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp%read_16_disablecopyonread_variable_17^Read_16/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_32IdentityRead_16/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*(
_output_shapes
:��k
Read_17/DisableCopyOnReadDisableCopyOnRead%read_17_disablecopyonread_variable_16*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp%read_17_disablecopyonread_variable_16^Read_17/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_34IdentityRead_17/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_18/DisableCopyOnReadDisableCopyOnRead%read_18_disablecopyonread_variable_15*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp%read_18_disablecopyonread_variable_15^Read_18/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_36IdentityRead_18/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*(
_output_shapes
:��k
Read_19/DisableCopyOnReadDisableCopyOnRead%read_19_disablecopyonread_variable_14*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp%read_19_disablecopyonread_variable_14^Read_19/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_38IdentityRead_19/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_20/DisableCopyOnReadDisableCopyOnRead%read_20_disablecopyonread_variable_13*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp%read_20_disablecopyonread_variable_13^Read_20/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_40IdentityRead_20/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*(
_output_shapes
:��k
Read_21/DisableCopyOnReadDisableCopyOnRead%read_21_disablecopyonread_variable_12*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp%read_21_disablecopyonread_variable_12^Read_21/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_42IdentityRead_21/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:�k
Read_22/DisableCopyOnReadDisableCopyOnRead%read_22_disablecopyonread_variable_11*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp%read_22_disablecopyonread_variable_11^Read_22/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_44IdentityRead_22/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*(
_output_shapes
:��k
Read_23/DisableCopyOnReadDisableCopyOnRead%read_23_disablecopyonread_variable_10*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp%read_23_disablecopyonread_variable_10^Read_23/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_46IdentityRead_23/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_24/DisableCopyOnReadDisableCopyOnRead$read_24_disablecopyonread_variable_9*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp$read_24_disablecopyonread_variable_9^Read_24/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_48IdentityRead_24/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*(
_output_shapes
:��j
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_variable_8*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_variable_8^Read_25/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_50IdentityRead_25/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_26/DisableCopyOnReadDisableCopyOnRead$read_26_disablecopyonread_variable_7*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp$read_26_disablecopyonread_variable_7^Read_26/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0b
Identity_52IdentityRead_26/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��g
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��j
Read_27/DisableCopyOnReadDisableCopyOnRead$read_27_disablecopyonread_variable_6*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp$read_27_disablecopyonread_variable_6^Read_27/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_54IdentityRead_27/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:�j
Read_28/DisableCopyOnReadDisableCopyOnRead$read_28_disablecopyonread_variable_5*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp$read_28_disablecopyonread_variable_5^Read_28/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_56IdentityRead_28/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_29/DisableCopyOnReadDisableCopyOnRead$read_29_disablecopyonread_variable_4*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp$read_29_disablecopyonread_variable_4^Read_29/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0a
Identity_58IdentityRead_29/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@f
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@j
Read_30/DisableCopyOnReadDisableCopyOnRead$read_30_disablecopyonread_variable_3*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp$read_30_disablecopyonread_variable_3^Read_30/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_60IdentityRead_30/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:@j
Read_31/DisableCopyOnReadDisableCopyOnRead$read_31_disablecopyonread_variable_2*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp$read_31_disablecopyonread_variable_2^Read_31/DisableCopyOnRead*
_output_shapes
:*
dtype0	\
Identity_62IdentityRead_31/ReadVariableOp:value:0*
T0	*
_output_shapes
:a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0	*
_output_shapes
:j
Read_32/DisableCopyOnReadDisableCopyOnRead$read_32_disablecopyonread_variable_1*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp$read_32_disablecopyonread_variable_1^Read_32/DisableCopyOnRead*
_output_shapes

:@
*
dtype0`
Identity_64IdentityRead_32/ReadVariableOp:value:0*
T0*
_output_shapes

:@
e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:@
h
Read_33/DisableCopyOnReadDisableCopyOnRead"read_33_disablecopyonread_variable*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp"read_33_disablecopyonread_variable^Read_33/DisableCopyOnRead*
_output_shapes
:
*
dtype0\
Identity_66IdentityRead_33/ReadVariableOp:value:0*
T0*
_output_shapes
:
a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:
o
Read_34/DisableCopyOnReadDisableCopyOnRead)read_34_disablecopyonread_vgg_net_bias_28*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp)read_34_disablecopyonread_vgg_net_bias_28^Read_34/DisableCopyOnRead*
_output_shapes
: *
dtype0\
Identity_68IdentityRead_34/ReadVariableOp:value:0*
T0*
_output_shapes
: a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
: q
Read_35/DisableCopyOnReadDisableCopyOnRead+read_35_disablecopyonread_vgg_net_kernel_27*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp+read_35_disablecopyonread_vgg_net_kernel_27^Read_35/DisableCopyOnRead*&
_output_shapes
: @*
dtype0h
Identity_70IdentityRead_35/ReadVariableOp:value:0*
T0*&
_output_shapes
: @m
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*&
_output_shapes
: @o
Read_36/DisableCopyOnReadDisableCopyOnRead)read_36_disablecopyonread_vgg_net_bias_21*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp)read_36_disablecopyonread_vgg_net_bias_21^Read_36/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_72IdentityRead_36/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes	
:�q
Read_37/DisableCopyOnReadDisableCopyOnRead+read_37_disablecopyonread_vgg_net_kernel_20*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp+read_37_disablecopyonread_vgg_net_kernel_20^Read_37/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_74IdentityRead_37/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Read_38/DisableCopyOnReadDisableCopyOnRead+read_38_disablecopyonread_vgg_net_kernel_31*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp+read_38_disablecopyonread_vgg_net_kernel_31^Read_38/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_76IdentityRead_38/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*&
_output_shapes
:o
Read_39/DisableCopyOnReadDisableCopyOnRead)read_39_disablecopyonread_vgg_net_bias_27*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp)read_39_disablecopyonread_vgg_net_bias_27^Read_39/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_78IdentityRead_39/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@q
Read_40/DisableCopyOnReadDisableCopyOnRead+read_40_disablecopyonread_vgg_net_kernel_26*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp+read_40_disablecopyonread_vgg_net_kernel_26^Read_40/DisableCopyOnRead*&
_output_shapes
:@@*
dtype0h
Identity_80IdentityRead_40/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@q
Read_41/DisableCopyOnReadDisableCopyOnRead+read_41_disablecopyonread_vgg_net_kernel_23*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp+read_41_disablecopyonread_vgg_net_kernel_23^Read_41/DisableCopyOnRead*(
_output_shapes
:��*
dtype0j
Identity_82IdentityRead_41/ReadVariableOp:value:0*
T0*(
_output_shapes
:��o
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Read_42/DisableCopyOnReadDisableCopyOnRead)read_42_disablecopyonread_vgg_net_bias_20*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp)read_42_disablecopyonread_vgg_net_bias_20^Read_42/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_84IdentityRead_42/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:�o
Read_43/DisableCopyOnReadDisableCopyOnRead)read_43_disablecopyonread_vgg_net_bias_19*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp)read_43_disablecopyonread_vgg_net_bias_19^Read_43/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_86IdentityRead_43/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:�o
Read_44/DisableCopyOnReadDisableCopyOnRead)read_44_disablecopyonread_vgg_net_bias_18*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp)read_44_disablecopyonread_vgg_net_bias_18^Read_44/DisableCopyOnRead*
_output_shapes	
:�*
dtype0]
Identity_88IdentityRead_44/ReadVariableOp:value:0*
T0*
_output_shapes	
:�b
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes	
:�o
Read_45/DisableCopyOnReadDisableCopyOnRead)read_45_disablecopyonread_vgg_net_bias_17*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp)read_45_disablecopyonread_vgg_net_bias_17^Read_45/DisableCopyOnRead*
_output_shapes
:@*
dtype0\
Identity_90IdentityRead_45/ReadVariableOp:value:0*
T0*
_output_shapes
:@a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:@o
Read_46/DisableCopyOnReadDisableCopyOnRead)read_46_disablecopyonread_vgg_net_bias_16*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp)read_46_disablecopyonread_vgg_net_bias_16^Read_46/DisableCopyOnRead*
_output_shapes
:
*
dtype0\
Identity_92IdentityRead_46/ReadVariableOp:value:0*
T0*
_output_shapes
:
a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:
o
Read_47/DisableCopyOnReadDisableCopyOnRead)read_47_disablecopyonread_vgg_net_bias_31*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp)read_47_disablecopyonread_vgg_net_bias_31^Read_47/DisableCopyOnRead*
_output_shapes
:*
dtype0\
Identity_94IdentityRead_47/ReadVariableOp:value:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:q
Read_48/DisableCopyOnReadDisableCopyOnRead+read_48_disablecopyonread_vgg_net_kernel_30*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp+read_48_disablecopyonread_vgg_net_kernel_30^Read_48/DisableCopyOnRead*&
_output_shapes
:*
dtype0h
Identity_96IdentityRead_48/ReadVariableOp:value:0*
T0*&
_output_shapes
:m
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*&
_output_shapes
:q
Read_49/DisableCopyOnReadDisableCopyOnRead+read_49_disablecopyonread_vgg_net_kernel_25*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp+read_49_disablecopyonread_vgg_net_kernel_25^Read_49/DisableCopyOnRead*&
_output_shapes
:@@*
dtype0h
Identity_98IdentityRead_49/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@m
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@o
Read_50/DisableCopyOnReadDisableCopyOnRead)read_50_disablecopyonread_vgg_net_bias_26*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp)read_50_disablecopyonread_vgg_net_bias_26^Read_50/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_100IdentityRead_50/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:@q
Read_51/DisableCopyOnReadDisableCopyOnRead+read_51_disablecopyonread_vgg_net_kernel_19*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp+read_51_disablecopyonread_vgg_net_kernel_19^Read_51/DisableCopyOnRead*(
_output_shapes
:��*
dtype0k
Identity_102IdentityRead_51/ReadVariableOp:value:0*
T0*(
_output_shapes
:��q
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Read_52/DisableCopyOnReadDisableCopyOnRead)read_52_disablecopyonread_vgg_net_bias_23*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp)read_52_disablecopyonread_vgg_net_bias_23^Read_52/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_104IdentityRead_52/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes	
:�q
Read_53/DisableCopyOnReadDisableCopyOnRead+read_53_disablecopyonread_vgg_net_kernel_22*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp+read_53_disablecopyonread_vgg_net_kernel_22^Read_53/DisableCopyOnRead*(
_output_shapes
:��*
dtype0k
Identity_106IdentityRead_53/ReadVariableOp:value:0*
T0*(
_output_shapes
:��q
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Read_54/DisableCopyOnReadDisableCopyOnRead)read_54_disablecopyonread_vgg_net_bias_30*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp)read_54_disablecopyonread_vgg_net_bias_30^Read_54/DisableCopyOnRead*
_output_shapes
:*
dtype0]
Identity_108IdentityRead_54/ReadVariableOp:value:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:o
Read_55/DisableCopyOnReadDisableCopyOnRead)read_55_disablecopyonread_vgg_net_bias_29*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp)read_55_disablecopyonread_vgg_net_bias_29^Read_55/DisableCopyOnRead*
_output_shapes
: *
dtype0]
Identity_110IdentityRead_55/ReadVariableOp:value:0*
T0*
_output_shapes
: c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
: q
Read_56/DisableCopyOnReadDisableCopyOnRead+read_56_disablecopyonread_vgg_net_kernel_28*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp+read_56_disablecopyonread_vgg_net_kernel_28^Read_56/DisableCopyOnRead*&
_output_shapes
:  *
dtype0i
Identity_112IdentityRead_56/ReadVariableOp:value:0*
T0*&
_output_shapes
:  o
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*&
_output_shapes
:  o
Read_57/DisableCopyOnReadDisableCopyOnRead)read_57_disablecopyonread_vgg_net_bias_24*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp)read_57_disablecopyonread_vgg_net_bias_24^Read_57/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_114IdentityRead_57/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes	
:�o
Read_58/DisableCopyOnReadDisableCopyOnRead)read_58_disablecopyonread_vgg_net_bias_25*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp)read_58_disablecopyonread_vgg_net_bias_25^Read_58/DisableCopyOnRead*
_output_shapes
:@*
dtype0]
Identity_116IdentityRead_58/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:@q
Read_59/DisableCopyOnReadDisableCopyOnRead+read_59_disablecopyonread_vgg_net_kernel_29*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp+read_59_disablecopyonread_vgg_net_kernel_29^Read_59/DisableCopyOnRead*&
_output_shapes
: *
dtype0i
Identity_118IdentityRead_59/ReadVariableOp:value:0*
T0*&
_output_shapes
: o
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*&
_output_shapes
: q
Read_60/DisableCopyOnReadDisableCopyOnRead+read_60_disablecopyonread_vgg_net_kernel_24*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp+read_60_disablecopyonread_vgg_net_kernel_24^Read_60/DisableCopyOnRead*'
_output_shapes
:@�*
dtype0j
Identity_120IdentityRead_60/ReadVariableOp:value:0*
T0*'
_output_shapes
:@�p
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�o
Read_61/DisableCopyOnReadDisableCopyOnRead)read_61_disablecopyonread_vgg_net_bias_22*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp)read_61_disablecopyonread_vgg_net_bias_22^Read_61/DisableCopyOnRead*
_output_shapes	
:�*
dtype0^
Identity_122IdentityRead_61/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes	
:�q
Read_62/DisableCopyOnReadDisableCopyOnRead+read_62_disablecopyonread_vgg_net_kernel_21*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp+read_62_disablecopyonread_vgg_net_kernel_21^Read_62/DisableCopyOnRead*(
_output_shapes
:��*
dtype0k
Identity_124IdentityRead_62/ReadVariableOp:value:0*
T0*(
_output_shapes
:��q
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Read_63/DisableCopyOnReadDisableCopyOnRead+read_63_disablecopyonread_vgg_net_kernel_18*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp+read_63_disablecopyonread_vgg_net_kernel_18^Read_63/DisableCopyOnRead* 
_output_shapes
:
��*
dtype0c
Identity_126IdentityRead_63/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��i
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��q
Read_64/DisableCopyOnReadDisableCopyOnRead+read_64_disablecopyonread_vgg_net_kernel_17*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp+read_64_disablecopyonread_vgg_net_kernel_17^Read_64/DisableCopyOnRead*
_output_shapes
:	�@*
dtype0b
Identity_128IdentityRead_64/ReadVariableOp:value:0*
T0*
_output_shapes
:	�@h
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:	�@q
Read_65/DisableCopyOnReadDisableCopyOnRead+read_65_disablecopyonread_vgg_net_kernel_16*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp+read_65_disablecopyonread_vgg_net_kernel_16^Read_65/DisableCopyOnRead*
_output_shapes

:@
*
dtype0a
Identity_130IdentityRead_65/ReadVariableOp:value:0*
T0*
_output_shapes

:@
g
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes

:@
L

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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/0/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/1/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/2/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/3/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/4/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/5/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/6/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/7/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/8/.ATTRIBUTES/VARIABLE_VALUEB+_all_variables/9/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/10/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/11/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/12/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/13/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/14/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/15/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/16/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/17/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/18/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/19/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/20/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/21/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/22/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/23/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/24/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/25/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/26/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/27/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/28/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/29/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/30/.ATTRIBUTES/VARIABLE_VALUEB,_all_variables/31/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:C*
dtype0*�
value�B�CB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *Q
dtypesG
E2C		�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_132Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_133IdentityIdentity_132:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_133Identity_133:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=C9

_output_shapes
: 

_user_specified_nameConst:1B-
+
_user_specified_namevgg_net/kernel_16:1A-
+
_user_specified_namevgg_net/kernel_17:1@-
+
_user_specified_namevgg_net/kernel_18:1?-
+
_user_specified_namevgg_net/kernel_21:/>+
)
_user_specified_namevgg_net/bias_22:1=-
+
_user_specified_namevgg_net/kernel_24:1<-
+
_user_specified_namevgg_net/kernel_29:/;+
)
_user_specified_namevgg_net/bias_25:/:+
)
_user_specified_namevgg_net/bias_24:19-
+
_user_specified_namevgg_net/kernel_28:/8+
)
_user_specified_namevgg_net/bias_29:/7+
)
_user_specified_namevgg_net/bias_30:16-
+
_user_specified_namevgg_net/kernel_22:/5+
)
_user_specified_namevgg_net/bias_23:14-
+
_user_specified_namevgg_net/kernel_19:/3+
)
_user_specified_namevgg_net/bias_26:12-
+
_user_specified_namevgg_net/kernel_25:11-
+
_user_specified_namevgg_net/kernel_30:/0+
)
_user_specified_namevgg_net/bias_31://+
)
_user_specified_namevgg_net/bias_16:/.+
)
_user_specified_namevgg_net/bias_17:/-+
)
_user_specified_namevgg_net/bias_18:/,+
)
_user_specified_namevgg_net/bias_19:/++
)
_user_specified_namevgg_net/bias_20:1*-
+
_user_specified_namevgg_net/kernel_23:1)-
+
_user_specified_namevgg_net/kernel_26:/(+
)
_user_specified_namevgg_net/bias_27:1'-
+
_user_specified_namevgg_net/kernel_31:1&-
+
_user_specified_namevgg_net/kernel_20:/%+
)
_user_specified_namevgg_net/bias_21:1$-
+
_user_specified_namevgg_net/kernel_27:/#+
)
_user_specified_namevgg_net/bias_28:("$
"
_user_specified_name
Variable:*!&
$
_user_specified_name
Variable_1:* &
$
_user_specified_name
Variable_2:*&
$
_user_specified_name
Variable_3:*&
$
_user_specified_name
Variable_4:*&
$
_user_specified_name
Variable_5:*&
$
_user_specified_name
Variable_6:*&
$
_user_specified_name
Variable_7:*&
$
_user_specified_name
Variable_8:*&
$
_user_specified_name
Variable_9:+'
%
_user_specified_nameVariable_10:+'
%
_user_specified_nameVariable_11:+'
%
_user_specified_nameVariable_12:+'
%
_user_specified_nameVariable_13:+'
%
_user_specified_nameVariable_14:+'
%
_user_specified_nameVariable_15:+'
%
_user_specified_nameVariable_16:+'
%
_user_specified_nameVariable_17:+'
%
_user_specified_nameVariable_18:+'
%
_user_specified_nameVariable_19:+'
%
_user_specified_nameVariable_20:+'
%
_user_specified_nameVariable_21:+'
%
_user_specified_nameVariable_22:+'
%
_user_specified_nameVariable_23:+
'
%
_user_specified_nameVariable_24:+	'
%
_user_specified_nameVariable_25:+'
%
_user_specified_nameVariable_26:+'
%
_user_specified_nameVariable_27:+'
%
_user_specified_nameVariable_28:+'
%
_user_specified_nameVariable_29:+'
%
_user_specified_nameVariable_30:+'
%
_user_specified_nameVariable_31:+'
%
_user_specified_nameVariable_32:+'
%
_user_specified_nameVariable_33:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�L
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serve�
7
args_0-
serve_args_0:0���������@@<
output_00
StatefulPartitionedCall:0���������
tensorflow/serving/predict*�
serving_default�
A
args_07
serving_default_args_0:0���������@@>
output_02
StatefulPartitionedCall_1:0���������
tensorflow/serving/predict:�'
�
	variables
trainable_variables
non_trainable_variables
_all_variables
_misc_assets
	serve

signatures"
_generic_user_object
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
$28
%29
&30
'31
(32
)33"
trackable_list_wrapper
�
0
	1

2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 24
!25
"26
#27
%28
&29
(30
)31"
trackable_list_wrapper
.
$0
'1"
trackable_list_wrapper
�
*0
+1
,2
-3
.4
/5
06
17
28
39
410
511
612
713
814
915
:16
;17
<18
=19
>20
?21
@22
A23
B24
C25
D26
E27
F28
G29
H30
I31"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jtrace_02�
__inference___call___129766�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *%�"
 ����������@@zJtrace_0
7
	Kserve
Lserving_default"
signature_map
*:((2vgg_net/kernel
:(2vgg_net/bias
*:((2vgg_net/kernel
:(2vgg_net/bias
*:( (2vgg_net/kernel
: (2vgg_net/bias
*:(  (2vgg_net/kernel
: (2vgg_net/bias
*:( @(2vgg_net/kernel
:@(2vgg_net/bias
*:(@@(2vgg_net/kernel
:@(2vgg_net/bias
*:(@@(2vgg_net/kernel
:@(2vgg_net/bias
+:)@�(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
$:"
��(2vgg_net/kernel
:�(2vgg_net/bias
1:/	(2#seed_generator/seed_generator_state
#:!	�@(2vgg_net/kernel
:@(2vgg_net/bias
3:1	(2%seed_generator_1/seed_generator_state
": @
(2vgg_net/kernel
:
(2vgg_net/bias
: (2vgg_net/bias
*:( @(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
*:((2vgg_net/kernel
:@(2vgg_net/bias
*:(@@(2vgg_net/kernel
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
:�(2vgg_net/bias
:�(2vgg_net/bias
:@(2vgg_net/bias
:
(2vgg_net/bias
:(2vgg_net/bias
*:((2vgg_net/kernel
*:(@@(2vgg_net/kernel
:@(2vgg_net/bias
,:*��(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
:(2vgg_net/bias
: (2vgg_net/bias
*:(  (2vgg_net/kernel
:�(2vgg_net/bias
:@(2vgg_net/bias
*:( (2vgg_net/kernel
+:)@�(2vgg_net/kernel
:�(2vgg_net/bias
,:*��(2vgg_net/kernel
$:"
��(2vgg_net/kernel
#:!	�@(2vgg_net/kernel
": @
(2vgg_net/kernel
�B�
__inference___call___129766args_0"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
-__inference_signature_wrapper___call___129836args_0"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jargs_0
kwonlydefaults
 
annotations� *
 
�B�
-__inference_signature_wrapper___call___129905args_0"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs�

jargs_0
kwonlydefaults
 
annotations� *
 �
__inference___call___129766~ 	
 !"#%&()7�4
-�*
(�%
args_0���������@@
� "!�
unknown���������
�
-__inference_signature_wrapper___call___129836� 	
 !"#%&()A�>
� 
7�4
2
args_0(�%
args_0���������@@"3�0
.
output_0"�
output_0���������
�
-__inference_signature_wrapper___call___129905� 	
 !"#%&()A�>
� 
7�4
2
args_0(�%
args_0���������@@"3�0
.
output_0"�
output_0���������
