//! PyRAI Solana program for model distillation verification

use anchor_lang::prelude::*;
use std::str::FromStr;

declare_id!("your_program_id");

#[program]
pub mod pyrai {
    use super::*;
    
    pub fn store_model_metadata(
        ctx: Context<StoreMetadata>,
        model_id: String,
        metadata: String,
    ) -> Result<()> {
        let metadata_account = &mut ctx.accounts.metadata;
        metadata_account.model_id = model_id;
        metadata_account.metadata = metadata;
        Ok(())
    }
    
    pub fn verify_distillation(
        ctx: Context<VerifyDistillation>,
        teacher_id: String,
        student_id: String,
        results: String,
    ) -> Result<()> {
        let distillation = &mut ctx.accounts.distillation;
        distillation.teacher_id = teacher_id;
        distillation.student_id = student_id;
        distillation.results = results;
        Ok(())
    }
    
    pub fn track_data_provenance(
        ctx: Context<TrackProvenance>,
        data_id: String,
        provenance: String,
    ) -> Result<()> {
        let provenance_account = &mut ctx.accounts.provenance;
        provenance_account.data_id = data_id;
        provenance_account.provenance = provenance;
        Ok(())
    }
}

#[derive(Accounts)]
pub struct StoreMetadata<'info> {
    #[account(mut)]
    pub metadata: Account<'info, ModelMetadata>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct VerifyDistillation<'info> {
    #[account(mut)]
    pub distillation: Account<'info, DistillationResult>,
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct TrackProvenance<'info> {
    #[account(mut)]
    pub provenance: Account<'info, DataProvenance>,
    pub authority: Signer<'info>,
}

#[account]
pub struct ModelMetadata {
    pub model_id: String,
    pub metadata: String,
}

#[account]
pub struct DistillationResult {
    pub teacher_id: String,
    pub student_id: String,
    pub results: String,
}

#[account]
pub struct DataProvenance {
    pub data_id: String,
    pub provenance: String,
} 