
	.text
.globl rnd_arr
	.type	rnd_arr, @function
rnd_arr:
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
	.size	rnd_arr, .-rnd_arr
.globl rnd_zero
	.type	rnd_zero, @function
rnd_zero:
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
	.size	rnd_zero, .-rnd_zero
.globl rnd_plinf
	.type	rnd_plinf, @function
rnd_plinf:
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
	.size	rnd_plinf, .-rnd_plinf
.globl rnd_moinf
	.type	rnd_moinf, @function
rnd_moinf:
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
	.size	rnd_moinf, .-rnd_moinf
.globl rnd_switch
	.type	rnd_switch, @function
rnd_switch:
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
	.size	rnd_switch, .-rnd_switch
	.ident	"GCC: (Ubuntu 4.4.1-4ubuntu9) 4.4.1"
	.section	.note.GNU-stack,"",@progbits
