import { useQuery } from '@tanstack/react-query';
import { Grid, Box } from '@mui/material';
import {
  Memory as MemoryIcon,
  Speed as SpeedIcon,
  Storage as StorageIcon,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { format } from 'date-fns';
import MetricsCard from '@/components/MetricsCard';
import MetricsChart from '@/components/MetricsChart';
import { trainingApi } from '@/api/client';

export default function Dashboard() {
  // Fetch system metrics
  const { data: systemMetrics } = useQuery({
    queryKey: ['systemMetrics'],
    queryFn: () => trainingApi.getSystemMetrics(),
    refetchInterval: 5000,
  });

  // Fetch active jobs
  const { data: activeJobs } = useQuery({
    queryKey: ['jobs', 'running'],
    queryFn: () => trainingApi.listJobs('running'),
    refetchInterval: 5000,
  });

  // Prepare metrics data
  const latestMetrics = systemMetrics?.[systemMetrics.length - 1];

  // Prepare chart data
  const cpuData = {
    labels: systemMetrics?.map((m) =>
      format(new Date(m.timestamp), 'HH:mm:ss')
    ) || [],
    datasets: [
      {
        label: 'CPU Usage',
        data: systemMetrics?.map((m) => m.cpu_percent) || [],
        borderColor: '#2196f3',
        backgroundColor: 'rgba(33, 150, 243, 0.1)',
        fill: true,
      },
    ],
  };

  const memoryData = {
    labels: systemMetrics?.map((m) =>
      format(new Date(m.timestamp), 'HH:mm:ss')
    ) || [],
    datasets: [
      {
        label: 'Memory Usage',
        data: systemMetrics?.map((m) => m.memory_percent) || [],
        borderColor: '#f50057',
        backgroundColor: 'rgba(245, 0, 87, 0.1)',
        fill: true,
      },
    ],
  };

  return (
    <Box>
      <Grid container spacing={3}>
        {/* Metrics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="CPU Usage"
            value={`${latestMetrics?.cpu_percent.toFixed(1)}%`}
            icon={<SpeedIcon />}
            color="#2196f3"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Memory Usage"
            value={`${latestMetrics?.memory_percent.toFixed(1)}%`}
            icon={<MemoryIcon />}
            color="#f50057"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Disk Usage"
            value={`${latestMetrics?.disk_usage.toFixed(1)}%`}
            icon={<StorageIcon />}
            color="#4caf50"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricsCard
            title="Active Jobs"
            value={activeJobs?.total || 0}
            icon={<TimelineIcon />}
            color="#ff9800"
          />
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={6}>
          <MetricsChart title="CPU Usage Over Time" data={cpuData} />
        </Grid>
        <Grid item xs={12} md={6}>
          <MetricsChart title="Memory Usage Over Time" data={memoryData} />
        </Grid>
      </Grid>
    </Box>
  );
} 