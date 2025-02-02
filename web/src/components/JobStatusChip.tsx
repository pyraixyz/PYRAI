import { Chip } from '@mui/material';
import {
  PlayArrow as RunningIcon,
  Pause as PendingIcon,
  Check as CompletedIcon,
  Error as FailedIcon,
  Stop as InterruptedIcon,
} from '@mui/icons-material';

interface JobStatusChipProps {
  status: string;
}

const statusConfig: Record<
  string,
  { label: string; color: 'default' | 'primary' | 'success' | 'error' | 'warning'; icon: JSX.Element }
> = {
  pending: {
    label: 'Pending',
    color: 'default',
    icon: <PendingIcon />,
  },
  running: {
    label: 'Running',
    color: 'primary',
    icon: <RunningIcon />,
  },
  completed: {
    label: 'Completed',
    color: 'success',
    icon: <CompletedIcon />,
  },
  failed: {
    label: 'Failed',
    color: 'error',
    icon: <FailedIcon />,
  },
  interrupted: {
    label: 'Interrupted',
    color: 'warning',
    icon: <InterruptedIcon />,
  },
};

export default function JobStatusChip({ status }: JobStatusChipProps) {
  const config = statusConfig[status.toLowerCase()] || statusConfig.failed;

  return (
    <Chip
      label={config.label}
      color={config.color}
      icon={config.icon}
      size="small"
    />
  );
} 