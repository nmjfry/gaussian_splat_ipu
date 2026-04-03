	.text
	.allow_optimizations
	.file	"3"
                                        # Start of file scope inline assembly
	.section	.func_ptrs_def._ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_5ivec2ES5_,"",@progbits
	.long	_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_5ivec2ES5_
	.long	_ZNK5splat10Gaussian2D14getBoundingBoxEv
	.long	_ZNK5splat10Gaussian2D6insideEff
	.long	_ZN5splat10Gaussian2D12cacheTrigIdsEv

                                        # End of file scope inline assembly
	.section	.text.__runCodelet_Transform4x4,"ax",@progbits
	.globl	__runCodelet_Transform4x4       # -- Begin function __runCodelet_Transform4x4
	.p2align	2
	.type	__runCodelet_Transform4x4,@function
__runCodelet_Transform4x4:              # @__runCodelet_Transform4x4
	.worker
.Lfunc_begin0:
# %bb.0:
	setzi $m6, 112
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	add $m11, $m12, -112
	get $m0, 1
	and $m0, $m0, 7
	shl $m1, $m0, 2
	ld32 $m2, $m13, $m15, 5
	cmpult $m2, $m1, $m2
	brz $m2, .LBB0_3
# %bb.1:
	shl $m2, $m0, 4
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	ld32 $m3, $m13, $m15, 4
	st32 $azero, $m3, $m2, 1
	st32 $azero, $m3, $m2, 2
	st32 $azero, $m3, $m2, 3
	st32step $azero, $m3, $m2+=, 24
	add $m1, $m1, 24
	ld32 $m3, $m13, $m15, 5
	cmpult $m3, $m1, $m3
	brnz $m3, .LBB0_2
.LBB0_3:
	st32 $azero, $m13, $m15, 11
	st32 $azero, $m13, $m15, 10
	st32 $azero, $m13, $m15, 15
	st32 $azero, $m13, $m15, 14
	st32 $azero, $m13, $m15, 19
	st32 $azero, $m13, $m15, 18
	st32 $azero, $m13, $m15, 23
	st32 $azero, $m13, $m15, 22
	brnz $m0, .LBB0_7
# %bb.4:
	{
		ld32 $m0, $m13, $m15, 6
		or $a1, $azero, 1109393408
	}
	{
		ld32 $a2, $m0, $m15, 0
		or $a0, $azero, 1107296256
	}
	f32fromui32 $a2, $a2
	f32div $a3, $a2, $a1
	f32int $a3, $a3, 2
	f32mul $a1, $a3, $a1
	f32sub $a2, $a2, $a1
	or $a1, $azero, 1101004800
	f32v2mul $ap1, $ap1, $ap0
	f32int $a2, $a2, 2
	f32int $a3, $a3, 2
	{
		st64 $ap1, $m11, $m15, 3
		f32v2add $ap0, $ap1, $ap0
	}
	st64 $ap0, $m11, $m15, 4
	ld32 $m0, $m13, $m15, 3
	brz $m0, .LBB0_7
# %bb.5:
	mov	$m0, $m15
	mov	$m9, $m15
	add $m1, $m11, 40
	setzi $m2, _ZTVN5splat10Gaussian2DE+8
	st32 $m2, $m11, $m15, 4                 # 4-byte Folded Spill
	add $m2, $m1, 4
	st32 $m2, $m11, $m15, 3                 # 4-byte Folded Spill
	add $m2, $m11, 24
	st32 $m2, $m11, $m15, 2                 # 4-byte Folded Spill
	add $m2, $m1, 20
	st32 $m2, $m11, $m15, 1                 # 4-byte Folded Spill
	st32 $m1, $m11, $m15, 5                 # 4-byte Folded Spill
	add $m8, $m1, 48
	mov	$m7, $m0
.LBB0_6:                                # =>This Inner Loop Header: Depth=1
	ld32 $m1, $m11, $m15, 4                 # 4-byte Folded Reload
	st32 $m1, $m11, $m15, 10
	ld32 $m1, $m13, $m15, 2
	ld32 $m2, $m1, $m0, 3
	ld32 $m3, $m11, $m15, 3                 # 4-byte Folded Reload
	st32 $m2, $m3, $m15, 3
	ld32 $m2, $m1, $m0, 2
	st32 $m2, $m3, $m15, 2
	ld32 $m2, $m1, $m0, 1
	st32 $m2, $m3, $m15, 1
	ld32step $m2, $m1, $m7+=, 15
	st32 $m2, $m3, $m15, 0
	ld32 $m2, $m1, $m0, 4
	ld32 $m3, $m11, $m15, 1                 # 4-byte Folded Reload
	st32 $m2, $m3, $m15, 0
	ld32 $m2, $m1, $m0, 5
	st32 $m2, $m3, $m15, 1
	ld32 $m2, $m1, $m0, 6
	st32 $m2, $m3, $m15, 2
	ld32 $m2, $m1, $m0, 7
	st32 $m2, $m3, $m15, 3
	ld32 $m2, $m1, $m0, 11
	ld32 $m3, $m1, $m0, 10
	ld32 $m4, $m1, $m0, 9
	st32 $m2, $m8, $m15, 0
	ld32 $m2, $m1, $m0, 12
	st32 $m2, $m8, $m15, 1
	ld32 $m2, $m1, $m0, 13
	st32 $m2, $m8, $m15, 2
	ld32 $m0, $m1, $m0, 14
	st32 $m4, $m11, $m15, 20
	st32 $m0, $m8, $m15, 3
	st32 $m3, $m11, $m15, 21
	mov	$m0, $m13
	ld32 $m1, $m11, $m15, 5                 # 4-byte Folded Reload
	ld32 $m2, $m11, $m15, 2                 # 4-byte Folded Reload
	call $m10, _ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE
	add $m9, $m9, 15
	ld32 $m0, $m13, $m15, 3
	cmpult $m1, $m9, $m0
	mov	$m0, $m7
	brnz $m1, .LBB0_6
.LBB0_7:
	exitz $m15
.Lfunc_end0:
	.size	__runCodelet_Transform4x4, .Lfunc_end0-__runCodelet_Transform4x4
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_Transform4x4
	.long	.Lfunc_begin0
	.byte	112
	.section	.text.__runCodelet_Transform4x4,"ax",@progbits
                                        # -- End function
	.section	.text.__runCodelet_LoadMatrix,"ax",@progbits
	.globl	__runCodelet_LoadMatrix         # -- Begin function __runCodelet_LoadMatrix
	.p2align	2
	.type	__runCodelet_LoadMatrix,@function
__runCodelet_LoadMatrix:                # @__runCodelet_LoadMatrix
	.supervisor
.Lfunc_begin1:
# %bb.0:
	ld32 $m0, $m0, $m15, 0
	put 80, $m0
	ld128putcs 0
	ld128putcs 4
	ld128putcs 8
	ld128putcs 12
	put 80, $m0
	ld128putcs 18
	ld128putcs 22
	ld128putcs 26
	ld128putcs 30
	setzi $m0, 1
	br $m10
.Lfunc_end1:
	.size	__runCodelet_LoadMatrix, .Lfunc_end1-__runCodelet_LoadMatrix
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_LoadMatrix
	.long	.Lfunc_begin1
	.byte	0
	.section	.text.__runCodelet_LoadMatrix,"ax",@progbits
                                        # -- End function
	.section	.text.__runCodelet_Transform4x4_amp,"ax",@progbits
	.globl	__runCodelet_Transform4x4_amp   # -- Begin function __runCodelet_Transform4x4_amp
	.p2align	2
	.type	__runCodelet_Transform4x4_amp,@function
__runCodelet_Transform4x4_amp:          # @__runCodelet_Transform4x4_amp
	.worker
.Lfunc_begin2:
# %bb.0:
	get $m0, 1
	#APP

	setzi $a0, 8
	put	1, $a0

	#NO_APP
	shl $m0, $m0, 3
	and $m0, $m0, 56
	ld32 $m1, $m13, $m15, 1
	sub $m1, $m1, $m0
	setzi $m2, 43691
	sort4x16lo $m3, $m1, $m15
	shr $m4, $m1, 16
	mul $m5, $m3, $m2
	mul $m2, $m4, $m2
	shr $m5, $m5, 16
	add $m2, $m2, $m5
	setzi $m5, 43690
	sort4x16lo $m6, $m2, $m15
	mul $m3, $m3, $m5
	shr $m2, $m2, 16
	mul $m4, $m4, $m5
	add $m3, $m3, $m6
	add $m2, $m4, $m2
	shr $m3, $m3, 16
	add $m2, $m2, $m3
	ld32 $m3, $m13, $m15, 0
	shl $m0, $m0, 2
	ld32 $m4, $m13, $m15, 2
	cmpslt $m5, $m1, 0
	shr $m1, $m2, 5
	add $m2, $m3, $m0
	add $m3, $m4, $m0
	movnz	$m1, $m5, $m15
	setzi $m6, 21
	setzi $m7, 48
	#APP

	.allow_optimizations


	# Fill (inject 8 elements):
	ld64step $ap0, $m15, $m2+=, 1
	f32sisoamp $apzero, $a0, $apzero, 48

	{
		ld64step $ap0, $m15, $m2+=, 1
		f32sisoamp $apzero, $a1, $apzero, 49
	}

	f32sisoamp $apzero, $a0, $apzero, 50

	{
		ld64step $ap0, $m15, $m2+=, 1
		f32sisoamp $apzero, $a1, $apzero, 51
	}


	# Note we switch from using $a0:1 to using $a2:3 here to
	# free up more dual issue slots later:
	{
		ld64step $ap1, $m15, $m2+=, $m6
		f32sisoamp $apzero, $a0, $apzero, 52
	}


	# Pre-load first input pair before entering the loop.
	# (Note we switch back to loads into $a0:1 ready for the loop):
	{
		ld64step $ap0, $m15, $m2+=, 1
		f32sisoamp $apzero, $a1, $apzero, 53
	}


	# Optimised load/store instructions (ldst64pace below) require
	# triple packed addresses:
	{
		tapack $m4:5, $m2, $m15, $m3
		f32sisoamp $apzero, $a2, $apzero, 54
	}


	# Main loop (inject 8 and retrieve 8 elements per iteration):
	.p2align	3

	{
		rpt $m1, 7
		f32sisoamp $apzero, $a3, $apzero, 55
	}	# This is not part of the loop

LOOP_START0:

	{
		nop
		f32sisoamp $ap1, $a0, $apzero, 48
	}


	{
		ldst64pace $ap0, $ap1, $m4:5+=, $m15, 0
		f32sisoamp $apzero, $a1, $apzero, 49
	}


	{
		nop
		f32sisoamp $ap1, $a0, $apzero, 50
	}


	{
		ldst64pace $ap0, $ap1, $m4:5+=, $m15, 0
		f32sisoamp $apzero, $a1, $apzero, 51
	}


	{
		nop
		f32sisoamp $ap1, $a0, $apzero, 52
	}


	# Use stride specification to jump the packed read pointer to the worker's next chunk:
	{
		ldst64pace $ap0, $ap1, $m4:5+=, $m6, 1
		f32sisoamp $apzero, $a1, $apzero, 53
	}


	{
		nop
		f32sisoamp $ap1, $a0, $apzero, 54
	}


	# Use stride specification to jump the packed write pointer to the worker's next chunk:
	{
		ldst64pace $ap0, $ap1, $m4:5+=, $m6, 4
		f32sisoamp $apzero, $a1, $apzero, 55
	}	# At the end of the loop this is an over-read


	# Drain (retrieve and store the last 8 elements):
	f32sisoamp $ap1, $azero, $apzero, 48

	{
		st64pace $ap1, $m4:5+=, $m15, 0
		f32sisoamp $apzero, $azero, $apzero, 49
	}

	f32sisoamp $ap1, $azero, $apzero, 50

	{
		st64pace $ap1, $m4:5+=, $m15, 0
		f32sisoamp $apzero, $azero, $apzero, 51
	}

	f32sisoamp $ap1, $azero, $apzero, 52

	{
		st64pace $ap1, $m4:5+=, $m15, 0
		f32sisoamp $apzero, $a1, $apzero, 53
	}

	f32sisoamp $ap1, $azero, $apzero, 54
	st64pace $ap1, $m4:5+=, $m15, 0

	#NO_APP
	exitz $m15
.Lfunc_end2:
	.size	__runCodelet_Transform4x4_amp, .Lfunc_end2-__runCodelet_Transform4x4_amp
	.section	.stack_sizes,"o",@progbits,.text.__runCodelet_Transform4x4_amp
	.long	.Lfunc_begin2
	.byte	0
	.section	.text.__runCodelet_Transform4x4_amp,"ax",@progbits
                                        # -- End function
	.section	.text._ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE,"axG",@progbits,_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE,comdat
	.weak	_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE # -- Begin function _ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE
	.p2align	2
	.type	_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE,@function
_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE: # @_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE
	.worker
.Lfunc_begin3:
# %bb.0:
	add $m11, $m11, -72
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m8, $m11, $m15, 5                 # 4-byte Folded Spill
	st32 $m10, $m11, $m15, 4                # 4-byte Folded Spill
	st64 $ap3, $m11, $m15, 1                # 8-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	mov	$m7, $m1
	mov	$m8, $m0
	ld32 $m0, $m7, $m15, 0
	ld32 $a2, $m2, $m15, 2
	ld32 $a6, $m2, $m15, 0
	ld32 $a0, $m2, $m15, 1
	st32 $a0, $m11, $m15, 10                # 4-byte Folded Spill
	ld32 $m3, $m0, $m15, 0
	ld32 $a3, $m2, $m15, 3
	st64 $ap1, $m11, $m15, 6                # 8-byte Folded Spill
	add $m0, $m11, 56
	setzi $m10, .Ltmp0
	br $m3
.Ltmp0:
	ld32 $a4, $m11, $m15, 14
	{
		ld32 $a0, $m11, $m15, 16
		f32cmplt $a3, $a4, $a6
	}
	ld32 $a2, $m11, $m15, 15
	ld32 $a1, $m11, $m15, 17
	mov	$m0, $a3
	st32 $a6, $m11, $m15, 6                 # 4-byte Folded Spill
	brnz $m0, .LBB3_2
# %bb.1:
	mov	$a6, $a4
.LBB3_2:
	st32 $a6, $m11, $m15, 11                # 4-byte Folded Spill
	ld32 $a4, $m11, $m15, 10                # 4-byte Folded Reload
	f32cmplt $a5, $a2, $a4
	mov	$m0, $a5
	brnz $m0, .LBB3_4
# %bb.3:
	mov	$a4, $a2
.LBB3_4:
	ld64 $ap3, $m11, $m15, 6                # 8-byte Folded Reload
	{
		ld32 $m0, $m7, $m15, 0
		f32v2cmpge $ap1, $ap0, $ap3
	}
	{
		ld32 $a5, $m11, $m15, 11
		andc64 $ap0, $ap0, $ap1
	}
	{
		st32 $a5, $m11, $m15, 14
		and64 $ap1, $ap3, $ap1
	}
	{
		st32 $a4, $m11, $m15, 15
		or64 $ap0, $ap1, $ap0
	}
	st32 $a0, $m11, $m15, 16
	st32 $a1, $m11, $m15, 17
	ld32 $m1, $m0, $m15, 2
	setzi $m10, .Ltmp1
	mov	$m0, $m7
	br $m1
.Ltmp1:
	{
		ld32 $m0, $m8, $m15, 6
		setzi $a0, 629146
	}
	{
		ld32 $m0, $m0, $m15, 0
		or $a0, $a0, 1049624576
	}
	sort4x16lo $m1, $m0, $m15
	mul $m2, $m1, 21846
	shrs $m3, $m0, 16
	shr $m2, $m2, 16
	mul $m4, $m3, 21846
	add $m2, $m4, $m2
	sort4x16lo $m4, $m2, $m15
	mul $m1, $m1, 21845
	add $m1, $m1, $m4
	shrs $m2, $m2, 16
	mul $m3, $m3, 21845
	shrs $m1, $m1, 16
	add $m2, $m3, $m2
	add $m1, $m2, $m1
	shr $m2, $m1, 31
	add $m1, $m1, $m2
	mul $m1, $m1, 3
	sub $m0, $m0, $m1
	cmpeq $m1, $m0, 1
	shl $m1, $m1, 31
	cmpeq $m2, $m0, 0
	shrs $m0, $m1, 31
	{
		shl $m1, $m2, 31
		mov	$a1, $a0
	}
	st64 $ap0, $m11, $m15, 4                # 8-byte Folded Spill
	brz $m0, .LBB3_5
# %bb.6:
	{
		shrs $m1, $m1, 31
		mov	$a0, $azero
	}
	brz $m0, .LBB3_7
.LBB3_8:
	st32 $azero, $m11, $m15, 7              # 4-byte Folded Spill
	brz $m1, .LBB3_9
.LBB3_10:
	brz $m1, .LBB3_12
.LBB3_11:
	ld64 $ap0, $m11, $m15, 4                # 8-byte Folded Reload
	{
		bri .LBB3_13
		mov	$a1, $azero
	}
.LBB3_5:
	ld64 $ap0, $m11, $m15, 4                # 8-byte Folded Reload
	mov	$a1, $azero
	{
		st64 $ap0, $m11, $m15, 4
		mov	$a0, $azero
	}
	shrs $m1, $m1, 31
	brnz $m0, .LBB3_8
.LBB3_7:
	ld64 $ap0, $m11, $m15, 4                # 8-byte Folded Reload
	st32 $azero, $m11, $m15, 7              # 4-byte Folded Spill
	brnz $m1, .LBB3_10
.LBB3_9:
	st32 $a0, $m11, $m15, 7                 # 4-byte Folded Spill
	brnz $m1, .LBB3_11
.LBB3_12:
	ld64 $ap0, $m11, $m15, 4                # 8-byte Folded Reload
	mov	$a0, $azero
.LBB3_13:
	st64 $ap0, $m11, $m15, 4                # 8-byte Folded Spill
	ld32 $a0, $m11, $m15, 14
	ld32 $a1, $m11, $m15, 16
	{
		st32 $a0, $m11, $m15, 12
		f32cmplt $a0, $a0, $a1
	}
	mov	$m0, $a0
	brz $m0, .LBB3_22
# %bb.14:
	ld32 $a2, $m11, $m15, 15
	ld32 $a0, $m11, $m15, 17
	{
		bri .LBB3_15
		f32cmplt $a2, $a2, $a0
	}
.LBB3_20:                               #   in Loop: Header=BB3_15 Depth=1
	ld32 $a1, $m11, $m15, 16
.LBB3_21:                               #   in Loop: Header=BB3_15 Depth=1
	{
		ld32 $a3, $m11, $m15, 12
		or $a2, $azero, 1065353216
	}
	f32add $a3, $a3, $a2
	{
		st32 $a3, $m11, $m15, 12
		f32cmplt $a2, $a3, $a1
	}
.LBB3_15:                               # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_18 Depth 2
	and $a2, $a2, 1
	mov	$m0, $a2
	brz $m0, .LBB3_22
# %bb.16:                               #   in Loop: Header=BB3_15 Depth=1
	ld32 $a7, $m11, $m15, 15
	f32cmplt $a2, $a7, $a0
	mov	$m0, $a2
	brz $m0, .LBB3_21
# %bb.17:                               #   in Loop: Header=BB3_15 Depth=1
	ld32 $a0, $m11, $m15, 6                 # 4-byte Folded Reload
	ld32 $a1, $m11, $m15, 12                # 4-byte Folded Reload
	f32sub $a0, $a1, $a0
	f32int $a0, $a0, 2
	st32 $a0, $m11, $m15, 11                # 4-byte Folded Spill
	bri .LBB3_18
.LBB3_23:                               #   in Loop: Header=BB3_18 Depth=2
	{
		ld64 $ap2, $m11, $m15, 4
		mov	$a3, $azero
	}
	{
		ld32 $a4, $m11, $m15, 7
		f32v2add $ap0, $ap2, $ap0
	}
	f32add $a2, $a4, $a2
.LBB3_24:                               #   in Loop: Header=BB3_18 Depth=2
	ld32 $a4, $m1, $m2, 3
	{
		add $m0, $m1, $m2
		f32add $a3, $a4, $a3
	}
	{
		st32 $a1, $m0, $m15, 1
		or $a1, $azero, 1065353216
	}
	{
		st32 $a0, $m0, $m15, 0
		f32add $a7, $a7, $a1
	}
	st32 $a2, $m0, $m15, 2
	st32 $a3, $m0, $m15, 3
	ld32 $a0, $m11, $m15, 17
	f32cmplt $a1, $a7, $a0
	and $a1, $a1, 1
	mov	$m0, $a1
	brz $m0, .LBB3_20
.LBB3_18:                               #   Parent Loop BB3_15 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	ld32 $a0, $m11, $m15, 10                # 4-byte Folded Reload
	{
		ld32 $m0, $m7, $m15, 0
		f32sub $a0, $a7, $a0
	}
	{
		ld32 $m1, $m0, $m15, 1
		f32int $a6, $a0, 2
	}
	setzi $m10, .Ltmp2
	mov	$m0, $m7
	{
		ld32 $a0, $m11, $m15, 12
		mov	$a1, $a7
	}
	br $m1
.Ltmp2:
	{
		ld32 $m1, $m8, $m15, 4
		or $a0, $azero, 1107296256
	}
	{
		ld32 $a1, $m11, $m15, 11
		f32mul $a0, $a6, $a0
	}
	f32add $a0, $a0, $a1
	f32int $a0, $a0, 3
	f32toui32 $a0, $a0
	mov	$m2, $a0
	shl $m2, $m2, 4
	ld32 $a1, $m1, $m2, 1
	ld32 $a0, $m1, $m2, 0
	ld32 $a2, $m1, $m2, 2
	brz $m0, .LBB3_23
# %bb.19:                               #   in Loop: Header=BB3_18 Depth=2
	ld32 $a4, $m7, $m15, 5
	ld32 $a5, $m7, $m15, 6
	{
		ld32 $a3, $m7, $m15, 7
		f32v2add $ap0, $ap0, $ap2
	}
	{
		ld32 $a3, $m7, $m15, 8
		f32add $a2, $a2, $a3
	}
	bri .LBB3_24
.LBB3_22:
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld64 $ap3, $m11, $m15, 1                # 8-byte Folded Reload
	ld32 $m10, $m11, $m15, 4                # 4-byte Folded Reload
	ld32 $m8, $m11, $m15, 5                 # 4-byte Folded Reload
	add $m11, $m11, 72
	br $m10
.Lfunc_end3:
	.size	_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE, .Lfunc_end3-_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE
	.section	.stack_sizes,"Go",@progbits,_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE,comdat,.text._ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE
	.long	.Lfunc_begin3
	.byte	72
	.section	.text._ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE,"axG",@progbits,_ZN12Transform4x45splatERN5splat9PrimitiveERKNS0_8Bounds2fE,comdat
                                        # -- End function
	.section	.text._ZNK5splat10Gaussian2D14getBoundingBoxEv,"axG",@progbits,_ZNK5splat10Gaussian2D14getBoundingBoxEv,comdat
	.weak	_ZNK5splat10Gaussian2D14getBoundingBoxEv # -- Begin function _ZNK5splat10Gaussian2D14getBoundingBoxEv
	.p2align	2
	.type	_ZNK5splat10Gaussian2D14getBoundingBoxEv,@function
_ZNK5splat10Gaussian2D14getBoundingBoxEv: # @_ZNK5splat10Gaussian2D14getBoundingBoxEv
	.worker
.Lfunc_begin4:
# %bb.0:
	add $m11, $m11, -8
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st64 $ap3, $m11, $m15, 0                # 8-byte Folded Spill
	{
		ld32 $a0, $m1, $m15, 16
		or $a2, $azero, 1056964608
	}
	ld32 $a1, $m1, $m15, 10
	{
		ld32 $a3, $m1, $m15, 11
		f32mul $a5, $a1, $a2
	}
	{
		ld32 $a1, $m1, $m15, 17
		f32mul $a6, $a3, $a2
	}
	f32v2mul $ap1, $ap0, $ap0
	mov	$a4, $a3
	f32mul $a7, $a5, $a5
	mov	$a5, $a2
	{
		ld32 $a0, $m1, $m15, 1
		f32mul $a1, $a6, $a6
	}
	{
		ld32 $a1, $m1, $m15, 2
		f32v2mul $ap2, $a1:B, $ap2
	}
	f32v2mul $ap1, $a7:B, $ap1
	f32v2add $ap1, $ap1, $ap2
	f32sqrt $a2, $a2
	f32sqrt $a3, $a3
	f32v2sub $ap2, $ap0, $ap1
	{
		st32 $a5, $m0, $m15, 1
		f32v2add $ap0, $ap0, $ap1
	}
	st32 $a4, $m0, $m15, 0
	st32 $a1, $m0, $m15, 3
	st32 $a0, $m0, $m15, 2
	ld64 $ap3, $m11, $m15, 0                # 8-byte Folded Reload
	add $m11, $m11, 8
	br $m10
.Lfunc_end4:
	.size	_ZNK5splat10Gaussian2D14getBoundingBoxEv, .Lfunc_end4-_ZNK5splat10Gaussian2D14getBoundingBoxEv
	.section	.stack_sizes,"Go",@progbits,_ZNK5splat10Gaussian2D14getBoundingBoxEv,comdat,.text._ZNK5splat10Gaussian2D14getBoundingBoxEv
	.long	.Lfunc_begin4
	.byte	8
	.section	.text._ZNK5splat10Gaussian2D14getBoundingBoxEv,"axG",@progbits,_ZNK5splat10Gaussian2D14getBoundingBoxEv,comdat
                                        # -- End function
	.section	.text._ZNK5splat10Gaussian2D6insideEff,"axG",@progbits,_ZNK5splat10Gaussian2D6insideEff,comdat
	.weak	_ZNK5splat10Gaussian2D6insideEff # -- Begin function _ZNK5splat10Gaussian2D6insideEff
	.p2align	2
	.type	_ZNK5splat10Gaussian2D6insideEff,@function
_ZNK5splat10Gaussian2D6insideEff:       # @_ZNK5splat10Gaussian2D6insideEff
	.worker
.Lfunc_begin5:
# %bb.0:
	add $m11, $m11, -16
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st64 $ap3, $m11, $m15, 0                # 8-byte Folded Spill
	{
		ld32 $a2, $m0, $m15, 16
		or $a3, $azero, 1056964608
	}
	st32 $a3, $m11, $m15, 3                 # 4-byte Folded Spill
	ld32 $a3, $m0, $m15, 1
	{
		ld32 $a5, $m0, $m15, 2
		f32sub $a0, $a0, $a3
	}
	{
		ld32 $a3, $m0, $m15, 17
		f32sub $a4, $a1, $a5
	}
	{
		ld32 $a6, $m0, $m15, 10
		f32sub $a5, $azero, $a2
	}
	ld32 $a7, $m0, $m15, 11
	ld32 $a1, $m11, $m15, 3                 # 4-byte Folded Reload
	f32v2mul $ap3, $a1:B, $ap3
	f32v2mul $ap0, $a0:B, $ap1
	mov	$a2, $a5
	f32v2mul $ap1, $a4:B, $ap1
	mov	$a4, $a3
	mov	$a5, $a2
	f32v2add $ap1, $ap0, $ap2
	f32v2mul $ap0, $ap3, $ap3
	f32v2mul $ap1, $ap1, $ap1
	f32div $a1, $a3, $a1
	f32div $a0, $a2, $a0
	f32add $a0, $a0, $a1
	or $a1, $azero, 1065353216
	f32cmple $a0, $a0, $a1
	and $a0, $a0, 1
	mov	$m0, $a0
	ld64 $ap3, $m11, $m15, 0                # 8-byte Folded Reload
	add $m11, $m11, 16
	br $m10
.Lfunc_end5:
	.size	_ZNK5splat10Gaussian2D6insideEff, .Lfunc_end5-_ZNK5splat10Gaussian2D6insideEff
	.section	.stack_sizes,"Go",@progbits,_ZNK5splat10Gaussian2D6insideEff,comdat,.text._ZNK5splat10Gaussian2D6insideEff
	.long	.Lfunc_begin5
	.byte	16
	.section	.text._ZNK5splat10Gaussian2D6insideEff,"axG",@progbits,_ZNK5splat10Gaussian2D6insideEff,comdat
                                        # -- End function
	.section	.text._ZN5splat10Gaussian2D12cacheTrigIdsEv,"axG",@progbits,_ZN5splat10Gaussian2D12cacheTrigIdsEv,comdat
	.weak	_ZN5splat10Gaussian2D12cacheTrigIdsEv # -- Begin function _ZN5splat10Gaussian2D12cacheTrigIdsEv
	.p2align	2
	.type	_ZN5splat10Gaussian2D12cacheTrigIdsEv,@function
_ZN5splat10Gaussian2D12cacheTrigIdsEv:  # @_ZN5splat10Gaussian2D12cacheTrigIdsEv
	.worker
.Lfunc_begin6:
# %bb.0:
	add $m11, $m11, -16
	sub $m6, $m12, $m11
	cmpult $m6, $m6, __worker_stack_max_growth_plus_one
	brz $m6, _stackoverflow
	st32 $m10, $m11, $m15, 3                # 4-byte Folded Spill
	st32 $a6, $m11, $m15, 2                 # 4-byte Folded Spill
	st32 $m7, $m11, $m15, 1                 # 4-byte Folded Spill
	mov	$m7, $m0
	ld32 $a6, $m7, $m15, 15
	mov	$a0, $a6
	call $m10, cosf
	{
		st32 $a0, $m7, $m15, 16
		mov	$a0, $a6
	}
	call $m10, sinf
	st32 $a0, $m7, $m15, 17
	ld32 $m7, $m11, $m15, 1                 # 4-byte Folded Reload
	ld32 $a6, $m11, $m15, 2                 # 4-byte Folded Reload
	ld32 $m10, $m11, $m15, 3                # 4-byte Folded Reload
	add $m11, $m11, 16
	br $m10
.Lfunc_end6:
	.size	_ZN5splat10Gaussian2D12cacheTrigIdsEv, .Lfunc_end6-_ZN5splat10Gaussian2D12cacheTrigIdsEv
	.section	.stack_sizes,"Go",@progbits,_ZN5splat10Gaussian2D12cacheTrigIdsEv,comdat,.text._ZN5splat10Gaussian2D12cacheTrigIdsEv
	.long	.Lfunc_begin6
	.byte	16
	.section	.text._ZN5splat10Gaussian2D12cacheTrigIdsEv,"axG",@progbits,_ZN5splat10Gaussian2D12cacheTrigIdsEv,comdat
                                        # -- End function
	.type	_ZTVN5splat10Gaussian2DE,@object # @_ZTVN5splat10Gaussian2DE
	.section	.rodata._ZTVN5splat10Gaussian2DE,"aG",@progbits,_ZTVN5splat10Gaussian2DE,comdat
	.weak	_ZTVN5splat10Gaussian2DE
	.p2align	2, 0x0
_ZTVN5splat10Gaussian2DE:
	.long	0
	.long	0
	.long	_ZNK5splat10Gaussian2D14getBoundingBoxEv
	.long	_ZNK5splat10Gaussian2D6insideEff
	.long	_ZN5splat10Gaussian2D12cacheTrigIdsEv
	.size	_ZTVN5splat10Gaussian2DE, 20

	.ident	"clang version 16.0.0 (ssh://git@phabricator.sourcevertex.net/diffusion/LLVMPROJECT/llvm-project.git 2fce0648f3c328b23a6cbc664fc0dd0630122212)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
