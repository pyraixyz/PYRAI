import { Box, LinearProgress, Typography } from '@mui/material';

interface JobProgressProps {
  progress: number;
  showLabel?: boolean;
}

export default function JobProgress({
  progress,
  showLabel = true,
}: JobProgressProps) {
  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: 'flex', alignItems: 'center' }}>
        <Box sx={{ width: '100%', mr: showLabel ? 1 : 0 }}>
          <LinearProgress
            variant="determinate"
            value={progress * 100}
            sx={{
              height: 8,
              borderRadius: 4,
              backgroundColor: 'rgba(255, 255, 255, 0.1)',
              '& .MuiLinearProgress-bar': {
                borderRadius: 4,
              },
            }}
          />
        </Box>
        {showLabel && (
          <Box sx={{ minWidth: 35 }}>
            <Typography variant="body2" color="text.secondary">
              {`${Math.round(progress * 100)}%`}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  );
} 