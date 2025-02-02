import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Stack,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import { systemApi } from '@/api/client';
import MetricsChart from '@/components/MetricsChart';

export default function SystemMetrics() {
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h'>('1h');

  // Fetch current metrics
  const { data: currentMetrics } = useQuery({
    queryKey: ['metrics', 'current'],
    queryFn: systemApi.getCurrentMetrics,
    refetchInterval: autoRefresh ? 5000 : false,
  });

  // Fetch historical metrics
  const { data: historicalMetrics } = useQuery({
    queryKey: ['metrics', 'historical', timeRange],
    queryFn: () => systemApi.getHistoricalMetrics(timeRange),
    refetchInterval: autoRefresh ? 5000 : false,
  });

  if (!currentMetrics || !historicalMetrics) {
    return null;
  }

  // Prepare data for charts
  const cpuData = {
    labels: historicalMetrics.map((m) => format(new Date(m.timestamp), 'HH:mm')),
    datasets: [
      {
        label: 'CPU Usage (%)',
        data: historicalMetrics.map((m) => m.cpu_usage * 100),
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const memoryData = {
    labels: historicalMetrics.map((m) => format(new Date(m.timestamp), 'HH:mm')),
    datasets: [
      {
        label: 'Memory Usage (%)',
        data: historicalMetrics.map((m) => m.memory_usage * 100),
        borderColor: 'rgb(255, 99, 132)',
        tension: 0.1,
      },
    ],
  };

  const diskData = {
    labels: historicalMetrics.map((m) => format(new Date(m.timestamp), 'HH:mm')),
    datasets: [
      {
        label: 'Disk Usage (%)',
        data: historicalMetrics.map((m) => m.disk_usage * 100),
        borderColor: 'rgb(153, 102, 255)',
        tension: 0.1,
      },
    ],
  };

  const gpuData = {
    labels: historicalMetrics.map((m) => format(new Date(m.timestamp), 'HH:mm')),
    datasets: currentMetrics.gpu_metrics.map((gpu, index) => ({
      label: `GPU ${index} Usage (%)`,
      data: historicalMetrics.map((m) => m.gpu_metrics[index].usage * 100),
      borderColor: `hsl(${(index * 137.5) % 360}, 70%, 50%)`,
      tension: 0.1,
    })),
  };

  return (
    <Box>
      {/* Header */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <Typography variant="h5">System Metrics</Typography>
        <Stack direction="row" spacing={2}>
          <Stack direction="row" spacing={1}>
            <Button
              variant={timeRange === '1h' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTimeRange('1h')}
            >
              1h
            </Button>
            <Button
              variant={timeRange === '6h' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTimeRange('6h')}
            >
              6h
            </Button>
            <Button
              variant={timeRange === '24h' ? 'contained' : 'outlined'}
              size="small"
              onClick={() => setTimeRange('24h')}
            >
              24h
            </Button>
          </Stack>
          <Button
            variant="outlined"
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            {autoRefresh ? 'Pause Updates' : 'Resume Updates'}
          </Button>
        </Stack>
      </Box>

      <Grid container spacing={3}>
        {/* Current Metrics */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Current Resource Usage
              </Typography>
              <TableContainer component={Paper} variant="outlined">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Resource</TableCell>
                      <TableCell align="right">Usage</TableCell>
                      <TableCell align="right">Total</TableCell>
                      <TableCell align="right">Usage %</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    <TableRow>
                      <TableCell>CPU</TableCell>
                      <TableCell align="right">
                        {(currentMetrics.cpu_usage * 100).toFixed(1)}%
                      </TableCell>
                      <TableCell align="right">
                        {currentMetrics.cpu_cores} Cores
                      </TableCell>
                      <TableCell align="right">
                        {((currentMetrics.cpu_usage / currentMetrics.cpu_cores) * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Memory</TableCell>
                      <TableCell align="right">
                        {(currentMetrics.memory_used / 1024).toFixed(1)} GB
                      </TableCell>
                      <TableCell align="right">
                        {(currentMetrics.memory_total / 1024).toFixed(1)} GB
                      </TableCell>
                      <TableCell align="right">
                        {(currentMetrics.memory_usage * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Disk</TableCell>
                      <TableCell align="right">
                        {(currentMetrics.disk_used / 1024).toFixed(1)} GB
                      </TableCell>
                      <TableCell align="right">
                        {(currentMetrics.disk_total / 1024).toFixed(1)} GB
                      </TableCell>
                      <TableCell align="right">
                        {(currentMetrics.disk_usage * 100).toFixed(1)}%
                      </TableCell>
                    </TableRow>
                    {currentMetrics.gpu_metrics.map((gpu, index) => (
                      <TableRow key={index}>
                        <TableCell>GPU {index}</TableCell>
                        <TableCell align="right">
                          {(gpu.memory_used / 1024).toFixed(1)} GB
                        </TableCell>
                        <TableCell align="right">
                          {(gpu.memory_total / 1024).toFixed(1)} GB
                        </TableCell>
                        <TableCell align="right">
                          {(gpu.usage * 100).toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Historical Charts */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                CPU Usage History
              </Typography>
              <MetricsChart
                data={cpuData}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      title: {
                        display: true,
                        text: 'Usage (%)',
                      },
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Memory Usage History
              </Typography>
              <MetricsChart
                data={memoryData}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      title: {
                        display: true,
                        text: 'Usage (%)',
                      },
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Disk Usage History
              </Typography>
              <MetricsChart
                data={diskData}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      title: {
                        display: true,
                        text: 'Usage (%)',
                      },
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                GPU Usage History
              </Typography>
              <MetricsChart
                data={gpuData}
                options={{
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 100,
                      title: {
                        display: true,
                        text: 'Usage (%)',
                      },
                    },
                  },
                }}
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
} 