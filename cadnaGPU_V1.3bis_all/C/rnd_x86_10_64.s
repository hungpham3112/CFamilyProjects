
	.text
.globl rnd_arr
	.type	_rnd_arr, @function
_rnd_arr:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rsp, %rbp
	.cfi_offset 6, -16
	.cfi_def_cfa_register 6
	fnstcw -2(%rbp);
	movzwl	-2(%rbp), %eax
	andb	$243, %ah
	movw	%ax, -2(%rbp)
	movzwl	-2(%rbp), %eax
	movw	%ax, -2(%rbp)
	fldcw -2(%rbp);
	stmxcsr -8(%rbp)
	movl	-8(%rbp), %eax
	andb	$159, %ah
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	movl	%eax, -8(%rbp)
	ldmxcsr -8(%rbp)
	leave
	ret
	.cfi_endproc
.LFE0:
	.size	_rnd_arr, .-_rnd_arr
.globl _rnd_zero
	.type	_rnd_zero, @function
_rnd_zero:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rsp, %rbp
	.cfi_offset 6, -16
	.cfi_def_cfa_register 6
	fnstcw -2(%rbp);
	movzwl	-2(%rbp), %eax
	andb	$243, %ah
	movw	%ax, -2(%rbp)
	movzwl	-2(%rbp), %eax
	orb	$12, %ah
	movw	%ax, -2(%rbp)
	fldcw -2(%rbp);
	stmxcsr -8(%rbp)
	movl	-8(%rbp), %eax
	andb	$159, %ah
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	orb	$96, %ah
	movl	%eax, -8(%rbp)
	ldmxcsr -8(%rbp)
	leave
	ret
	.cfi_endproc
.LFE1:
	.size	_rnd_zero, .-_rnd_zero
.globl _rnd_plinf
	.type	_rnd_plinf, @function
_rnd_plinf:
.LFB2:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rsp, %rbp
	.cfi_offset 6, -16
	.cfi_def_cfa_register 6
	fnstcw -2(%rbp);
	movzwl	-2(%rbp), %eax
	andb	$243, %ah
	movw	%ax, -2(%rbp)
	movzwl	-2(%rbp), %eax
	orb	$8, %ah
	movw	%ax, -2(%rbp)
	fldcw -2(%rbp);
	stmxcsr -8(%rbp)
	movl	-8(%rbp), %eax
	andb	$159, %ah
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	orb	$64, %ah
	movl	%eax, -8(%rbp)
	ldmxcsr -8(%rbp)
	leave
	ret
	.cfi_endproc
.LFE2:
	.size	_rnd_plinf, .-_rnd_plinf
.globl _rnd_moinf
	.type	_rnd_moinf, @function
_rnd_moinf:
.LFB3:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rsp, %rbp
	.cfi_offset 6, -16
	.cfi_def_cfa_register 6
	fnstcw -2(%rbp);
	movzwl	-2(%rbp), %eax
	andb	$243, %ah
	movw	%ax, -2(%rbp)
	movzwl	-2(%rbp), %eax
	orb	$4, %ah
	movw	%ax, -2(%rbp)
	fldcw -2(%rbp);
	stmxcsr -8(%rbp)
	movl	-8(%rbp), %eax
	andb	$159, %ah
	movl	%eax, -8(%rbp)
	movl	-8(%rbp), %eax
	orb	$32, %ah
	movl	%eax, -8(%rbp)
	ldmxcsr -8(%rbp)
	leave
	ret
	.cfi_endproc
.LFE3:
	.size	_rnd_moinf, .-_rnd_moinf
.globl _rnd_switch
	.type	_rnd_switch, @function
_rnd_switch:
.LFB4:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	movq	%rsp, %rbp
	.cfi_offset 6, -16
	.cfi_def_cfa_register 6
	fnstcw -2(%rbp);
	movzwl	-2(%rbp), %eax
	andw	$1024, %ax
	movw	%ax, -4(%rbp)
	movzwl	-4(%rbp), %eax
	addl	%eax, %eax
	movzwl	-4(%rbp), %edx
	notl	%edx
	andw	$1024, %dx
	orl	%edx, %eax
	movw	%ax, -6(%rbp)
	movzwl	-2(%rbp), %eax
	andb	$243, %ah
	movw	%ax, -2(%rbp)
	movzwl	-2(%rbp), %eax
	orw	-6(%rbp), %ax
	movw	%ax, -2(%rbp)
	fldcw -2(%rbp);
	stmxcsr -12(%rbp)
	movl	-12(%rbp), %eax
	andb	$159, %ah
	movl	%eax, -12(%rbp)
	movzwl	-6(%rbp), %eax
	leal	0(,%rax,8), %edx
	movl	-12(%rbp), %eax
	orl	%edx, %eax
	movl	%eax, -12(%rbp)
	ldmxcsr -12(%rbp)
	leave
	ret
	.cfi_endproc
.LFE4:
	.size	_rnd_switch, .-_rnd_switch
	.ident	"GCC: (Ubuntu 4.4.1-4ubuntu9) 4.4.1"
	.section	.note.GNU-stack,"",@progbits
