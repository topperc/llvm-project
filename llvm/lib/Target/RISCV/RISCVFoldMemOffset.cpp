//===- RISCVFoldMemOffset.cpp - Fold ADDI into memory offsets ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass tries to turn ADDIs into removable copies by folding their offset
// into later memory instructions.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-fold-mem-offset"
#define RISCV_FOLD_MEM_OFFSET_NAME "RISC-V Fold Memory Offset"

namespace {

class RISCVFoldMemOffset : public MachineFunctionPass {
  const MachineRegisterInfo *MRI;
public:
  static char ID;

  RISCVFoldMemOffset() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  bool allUsersFoldable(MachineInstr &MI, int64_t Offset,
                        SmallVectorImpl<std::pair<MachineInstr *, int64_t>> &FoldableInstrs,
                        SmallPtrSetImpl<MachineInstr *> &Visited);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_FOLD_MEM_OFFSET_NAME; }
};

} // end anonymous namespace

char RISCVFoldMemOffset::ID = 0;
INITIALIZE_PASS(RISCVFoldMemOffset, DEBUG_TYPE, RISCV_FOLD_MEM_OFFSET_NAME, false,
                false)

FunctionPass *llvm::createRISCVFoldMemOffsetPass() {
  return new RISCVFoldMemOffset();
}

bool RISCVFoldMemOffset::allUsersFoldable(MachineInstr &MI, int64_t Offset,
                                          SmallVectorImpl<std::pair<MachineInstr *, int64_t>> &FoldableInstrs,
                                          SmallPtrSetImpl<MachineInstr *> &Visited) {
  if (!Visited.insert(&MI).second)
    return false;

  for (auto &UserOp : MRI->use_nodbg_operands(MI.getOperand(0).getReg())) {
    MachineInstr *UserMI = UserOp.getParent();
    unsigned OpIdx = UserOp.getOperandNo();

    int64_t NewOffset = Offset;
    switch (UserMI->getOpcode()) {
    default:
      return false;
    case RISCV::ADD:
    case RISCV::ADDI:
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    case RISCV::SUB: {
      if (OpIdx == 2)
        NewOffset = -static_cast<uint64_t>(NewOffset);
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    }
    case RISCV::SLLI: {
      unsigned ShAmt = UserMI->getOperand(2).getImm();
      NewOffset = static_cast<uint64_t>(NewOffset) << ShAmt;
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    }
    case RISCV::SH1ADD:
      if (OpIdx == 1)
        NewOffset = static_cast<uint64_t>(NewOffset) << 1;
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    case RISCV::SH2ADD:
      if (OpIdx == 1)
        NewOffset = static_cast<uint64_t>(NewOffset) << 2;
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    case RISCV::SH3ADD:
      if (OpIdx == 1)
        NewOffset = static_cast<uint64_t>(NewOffset) << 3;
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    case RISCV::ADD_UW:
    case RISCV::SH1ADD_UW:
    case RISCV::SH2ADD_UW:
    case RISCV::SH3ADD_UW:
      // We can't sink addi through the zero extended input.
      if (OpIdx != 2)
        return false;
      if (!allUsersFoldable(*UserMI, NewOffset, FoldableInstrs, Visited))
        return false;
      break;
    case RISCV::LB:
    case RISCV::LBU:
    case RISCV::SB:
    case RISCV::LH:
    case RISCV::LH_INX:
    case RISCV::LHU:
    case RISCV::FLH:
    case RISCV::SH:
    case RISCV::SH_INX:
    case RISCV::FSH:
    case RISCV::LW:
    case RISCV::LW_INX:
    case RISCV::LWU:
    case RISCV::FLW:
    case RISCV::SW:
    case RISCV::SW_INX:
    case RISCV::FSW:
    case RISCV::LD:
    case RISCV::FLD:
    case RISCV::SD:
    case RISCV::FSD: {
      // FIXME: Share implementation with RISCVInstrInfo::canFoldIntoAddrMode.

      // Can't fold into store value.
      if (OpIdx == 0)
        return false;

      if (!UserMI->getOperand(2).isImm())
        return false;

      int64_t LocalOffset = UserMI->getOperand(2).getImm();
      assert(isInt<12>(LocalOffset));
      int64_t CombinedOffset = (uint64_t)LocalOffset + (uint64_t)NewOffset;
      if (!isInt<12>(CombinedOffset))
        return false;

      FoldableInstrs.emplace_back(UserMI, CombinedOffset);
      break;
    }
    }
  }

  return true;
}


bool RISCVFoldMemOffset::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *ST.getInstrInfo();

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      if (MI.getOpcode() != RISCV::ADDI)
        continue;

      // We only want to optimize register ADDIs.
      if (!MI.getOperand(1).isReg() ||
          !MI.getOperand(2).isImm())
        continue;

      int64_t Offset = MI.getOperand(2).getImm();
      assert(isInt<12>(Offset));

      SmallVector<std::pair<MachineInstr *, int64_t>> FoldableInstrs;
      SmallPtrSet<MachineInstr *, 8> Visited;

      if (!allUsersFoldable(MI, Offset, FoldableInstrs, Visited))
        continue;

      if (FoldableInstrs.empty())
        continue;

      // We can fold this ADDI.
      // Rewrite all the instructions.
      for (auto [MemMI, NewOffset] : FoldableInstrs)
        MemMI->getOperand(2).setImm(NewOffset);

      // Replace ADDI with a copy.
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(RISCV::COPY))
          .add(MI.getOperand(0))
          .add(MI.getOperand(1));
      MI.eraseFromParent();
    }
  }

  return MadeChange;
}
