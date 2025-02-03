import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Stack,
  Button,
  ButtonGroup,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Divider,
  Tooltip,
  IconButton,
} from '@mui/material';
import {
  Analytics as AnalyticsIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  DateRange as DateRangeIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format } from 'date-fns';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { modelApi } from '@/api/client';

interface TimeRange {
  start: Date;
  end: Date;
}

interface MetricCard {
  title: string;
  value: string | number;
  change?: number;
  icon: React.ReactNode;
}

interface ChartData {
  name: string;
  value: number;
}

function MetricCard({ title, value, change, icon }: MetricCard) {
  return (
    <Card>
      <CardContent>
        <Stack direction="row" spacing={2} alignItems="center">
          <Box sx={{ color: 'primary.main' }}>{icon}</Box>
          <Box>
            <Typography variant="h6" gutterBottom>
              {value}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {title}
            </Typography>
            {change !== undefined && (
              <Typography
                variant="caption"
                color={change >= 0 ? 'success.main' : 'error.main'}
              >
                {change >= 0 ? '+' : ''}
                {change}% vs last period
              </Typography>
            )}
          </Box>
        </Stack>
      </CardContent>
    </Card>
  );
}

export default function ModelAnalytics() {
  const { modelId } = useParams<{ modelId: string }>();
  const [timeRange, setTimeRange] = useState<'24h' | '7d' | '30d' | '90d'>('7d');
  const [metricType, setMetricType] = useState<
    'usage' | 'performance' | 'errors' | 'latency'
  >('usage');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch analytics data
  const { data: analytics, refetch: refetchAnalytics } = useQuery({
    queryKey: ['model', modelId, 'analytics', timeRange, metricType],
    queryFn: () =>
      modelApi.getAnalytics(modelId!, {
        timeRange,
        metricType,
      }),
    enabled: !!modelId,
  });

  if (!model || !analytics) {
    return null;
  }

  const {
    usage,
    performance,
    users,
    errors,
    latency,
    timeSeriesData,
    distributionData,
    geographicData,
  } = analytics;

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Analytics
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Usage analytics and performance metrics for {model.name}
        </Typography>
      </Box>

      {/* Controls */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: 3 }}
      >
        <ButtonGroup>
          <Button
            variant={timeRange === '24h' ? 'contained' : 'outlined'}
            onClick={() => setTimeRange('24h')}
          >
            24h
          </Button>
          <Button
            variant={timeRange === '7d' ? 'contained' : 'outlined'}
            onClick={() => setTimeRange('7d')}
          >
            7d
          </Button>
          <Button
            variant={timeRange === '30d' ? 'contained' : 'outlined'}
            onClick={() => setTimeRange('30d')}
          >
            30d
          </Button>
          <Button
            variant={timeRange === '90d' ? 'contained' : 'outlined'}
            onClick={() => setTimeRange('90d')}
          >
            90d
          </Button>
        </ButtonGroup>

        <Stack direction="row" spacing={2}>
          <FormControl size="small" sx={{ minWidth: 200 }}>
            <InputLabel>Metric Type</InputLabel>
            <Select
              value={metricType}
              label="Metric Type"
              onChange={(e) =>
                setMetricType(
                  e.target.value as 'usage' | 'performance' | 'errors' | 'latency'
                )
              }
            >
              <MenuItem value="usage">Usage Statistics</MenuItem>
              <MenuItem value="performance">Performance Metrics</MenuItem>
              <MenuItem value="errors">Error Analysis</MenuItem>
              <MenuItem value="latency">Latency Distribution</MenuItem>
            </Select>
          </FormControl>

          <Tooltip title="Refresh">
            <IconButton onClick={() => refetchAnalytics()}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Export Data">
            <IconButton>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        </Stack>
      </Stack>

      {/* Metric Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Total API Calls"
            value={usage.totalCalls.toLocaleString()}
            change={usage.callsChange}
            icon={<AnalyticsIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Users"
            value={users.activeUsers.toLocaleString()}
            change={users.activeUsersChange}
            icon={<AnalyticsIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Error Rate"
            value={`${errors.errorRate.toFixed(2)}%`}
            change={-errors.errorRateChange}
            icon={<AnalyticsIcon />}
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Avg Response Time"
            value={`${latency.avgResponseTime.toFixed(0)}ms`}
            change={-latency.responseTimeChange}
            icon={<AnalyticsIcon />}
          />
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Time Series Chart */}
        <Grid item xs={12}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              {metricType === 'usage' && 'API Calls Over Time'}
              {metricType === 'performance' && 'Performance Trends'}
              {metricType === 'errors' && 'Error Rate Trends'}
              {metricType === 'latency' && 'Response Time Trends'}
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={timeSeriesData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="timestamp" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#8884d8"
                    activeDot={{ r: 8 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Distribution Chart */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              {metricType === 'usage' && 'Usage Distribution'}
              {metricType === 'performance' && 'Performance Distribution'}
              {metricType === 'errors' && 'Error Distribution'}
              {metricType === 'latency' && 'Latency Distribution'}
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>

        {/* Geographic Chart */}
        <Grid item xs={12} md={6}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Geographic Distribution
            </Typography>
            <Box sx={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={geographicData}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    fill="#8884d8"
                    label
                  />
                  <RechartsTooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
} 