; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 4
; RUN: opt < %s -passes=newgvn -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

; rdar://12801584
; Value of %shouldExit can be changed by RunInMode.
; Make sure we do not replace load %shouldExit in while.cond.backedge
; with a phi node where the value from while.body is 0.
define i32 @test() nounwind ssp {
; CHECK-LABEL: define i32 @test(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SHOULDEXIT:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[TASKSIDLE:%.*]] = alloca i32, align 4
; CHECK-NEXT:    store i32 0, ptr [[SHOULDEXIT]], align 4
; CHECK-NEXT:    store i32 0, ptr [[TASKSIDLE]], align 4
; CHECK-NEXT:    call void @CTestInitialize(ptr [[TASKSIDLE]]) #[[ATTR1:[0-9]+]]
; CHECK-NEXT:    [[TMP0:%.*]] = load i32, ptr [[SHOULDEXIT]], align 4
; CHECK-NEXT:    [[CMP1:%.*]] = icmp eq i32 [[TMP0]], 0
; CHECK-NEXT:    br i1 [[CMP1]], label [[WHILE_BODY_LR_PH:%.*]], label [[WHILE_END:%.*]]
; CHECK:       while.body.lr.ph:
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.body:
; CHECK-NEXT:    call void @RunInMode(i32 100) #[[ATTR1]]
; CHECK-NEXT:    [[TMP1:%.*]] = load i32, ptr [[TASKSIDLE]], align 4
; CHECK-NEXT:    [[TOBOOL:%.*]] = icmp eq i32 [[TMP1]], 0
; CHECK-NEXT:    br i1 [[TOBOOL]], label [[WHILE_COND_BACKEDGE:%.*]], label [[IF_THEN:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    store i32 0, ptr [[TASKSIDLE]], align 4
; CHECK-NEXT:    call void @TimerCreate(ptr [[SHOULDEXIT]]) #[[ATTR1]]
; CHECK-NEXT:    br label [[WHILE_COND_BACKEDGE]]
; CHECK:       while.cond.backedge:
; CHECK-NEXT:    [[TMP2:%.*]] = load i32, ptr [[SHOULDEXIT]], align 4
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i32 [[TMP2]], 0
; CHECK-NEXT:    br i1 [[CMP]], label [[WHILE_BODY]], label [[WHILE_COND_WHILE_END_CRIT_EDGE:%.*]]
; CHECK:       while.cond.while.end_crit_edge:
; CHECK-NEXT:    br label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    ret i32 0
;
entry:
  %shouldExit = alloca i32, align 4
  %tasksIdle = alloca i32, align 4
  store i32 0, ptr %shouldExit, align 4
  store i32 0, ptr %tasksIdle, align 4
  call void @CTestInitialize(ptr %tasksIdle) nounwind
  %0 = load i32, ptr %shouldExit, align 4
  %cmp1 = icmp eq i32 %0, 0
  br i1 %cmp1, label %while.body.lr.ph, label %while.end

while.body.lr.ph:
  br label %while.body

while.body:
  call void @RunInMode(i32 100) nounwind
  %1 = load i32, ptr %tasksIdle, align 4
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %while.cond.backedge, label %if.then

if.then:
  store i32 0, ptr %tasksIdle, align 4
  call void @TimerCreate(ptr %shouldExit) nounwind
  br label %while.cond.backedge

while.cond.backedge:
  %2 = load i32, ptr %shouldExit, align 4
  %cmp = icmp eq i32 %2, 0
  br i1 %cmp, label %while.body, label %while.cond.while.end_crit_edge

while.cond.while.end_crit_edge:
  br label %while.end

while.end:
  ret i32 0
}
declare void @CTestInitialize(ptr)
declare void @RunInMode(i32)
declare void @TimerCreate(ptr)
