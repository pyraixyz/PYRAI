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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  IconButton,
  Chip,
  Autocomplete,
  TextField,
} from '@mui/material';
import {
  Compare as CompareIcon,
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { modelApi } from '@/api/client';

interface ComparisonMetric {
  name: string;
  type: 'accuracy' | 'loss' | 'latency' | 'memory' | 'custom';
  value: number;
  unit?: string;
  description?: string;
}

interface ModelVersion {
  id: string;
  name: string;
  version: string;
  metrics: ComparisonMetric[];
  created_at: string;
  created_by: {
    id: string;
    name: string;
  };
}

export default function ModelComparison() {
  const { modelId } = useParams<{ modelId: string }>();
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [comparisonType, setComparisonType] = useState<
    'versions' | 'models' | 'experiments'
  >('versions');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch model versions
  const { data: versions } = useQuery({
    queryKey: ['model', modelId, 'versions'],
    queryFn: () => modelApi.getModelVersions(modelId!),
    enabled: !!modelId,
  });

  // Fetch comparison data
  const { data: comparison, refetch: refetchComparison } = useQuery({
    queryKey: ['model', modelId, 'comparison', selectedVersions, selectedMetrics],
    queryFn: () =>
      modelApi.getComparison(modelId!, {
        versions: selectedVersions,
        metrics: selectedMetrics,
        type: comparisonType,
      }),
    enabled: !!modelId && selectedVersions.length > 0 && selectedMetrics.length > 0,
  });

  if (!model || !versions) {
    return null;
  }

  const handleAddVersion = (version: ModelVersion) => {
    if (selectedVersions.length < 5) {
      setSelectedVersions([...selectedVersions, version.id]);
    }
  };

  const handleRemoveVersion = (versionId: string) => {
    setSelectedVersions(selectedVersions.filter((id) => id !== versionId));
  };

  const getMetricColor = (type: string) => {
    switch (type) {
      case 'accuracy':
        return 'success';
      case 'loss':
        return 'error';
      case 'latency':
        return 'warning';
      case 'memory':
        return 'info';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Model Comparison
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Compare different versions or models to analyze their performance
        </Typography>
      </Box>

      {/* Controls */}
      <Stack spacing={3} sx={{ mb: 3 }}>
        <Stack direction="row" spacing={2} alignItems="center">
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Comparison Type</InputLabel>
            <Select
              value={comparisonType}
              label="Comparison Type"
              onChange={(e) =>
                setComparisonType(
                  e.target.value as 'versions' | 'models' | 'experiments'
                )
              }
            >
              <MenuItem value="versions">Version Comparison</MenuItem>
              <MenuItem value="models">Model Comparison</MenuItem>
              <MenuItem value="experiments">Experiment Comparison</MenuItem>
            </Select>
          </FormControl>

          <Tooltip title="Refresh">
            <IconButton onClick={() => refetchComparison()}>
              <RefreshIcon />
            </IconButton>
          </Tooltip>

          <Tooltip title="Export Data">
            <IconButton>
              <DownloadIcon />
            </IconButton>
          </Tooltip>
        </Stack>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Select Versions to Compare
              </Typography>
              <Stack spacing={2}>
                <Autocomplete
                  options={versions.filter(
                    (v) => !selectedVersions.includes(v.id)
                  )}
                  getOptionLabel={(option) => `${option.name} (${option.version})`}
                  renderInput={(params) => (
                    <TextField {...params} label="Add Version" />
                  )}
                  onChange={(_, value) => value && handleAddVersion(value)}
                />
                <Box>
                  {selectedVersions.map((versionId) => {
                    const version = versions.find((v) => v.id === versionId);
                    if (!version) return null;
                    return (
                      <Chip
                        key={version.id}
                        label={`${version.name} (${version.version})`}
                        onDelete={() => handleRemoveVersion(version.id)}
                        sx={{ mr: 1, mb: 1 }}
                      />
                    );
                  })}
                </Box>
              </Stack>
            </Paper>
          </Grid>

          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Select Metrics to Compare
              </Typography>
              <FormControl fullWidth>
                <Autocomplete
                  multiple
                  options={[
                    'accuracy',
                    'loss',
                    'latency',
                    'memory',
                    'throughput',
                    'error_rate',
                  ]}
                  value={selectedMetrics}
                  onChange={(_, value) => setSelectedMetrics(value)}
                  renderInput={(params) => (
                    <TextField {...params} label="Metrics" />
                  )}
                  renderTags={(value, getTagProps) =>
                    value.map((option, index) => (
                      <Chip
                        {...getTagProps({ index })}
                        key={option}
                        label={option}
                        color={getMetricColor(option)}
                      />
                    ))
                  }
                />
              </FormControl>
            </Paper>
          </Grid>
        </Grid>
      </Stack>

      {comparison && (
        <Grid container spacing={3}>
          {/* Metrics Table */}
          <Grid item xs={12}>
            <TableContainer component={Paper} variant="outlined">
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    {selectedVersions.map((versionId) => {
                      const version = versions.find((v) => v.id === versionId);
                      return (
                        <TableCell key={versionId} align="right">
                          {version?.name} ({version?.version})
                        </TableCell>
                      );
                    })}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {selectedMetrics.map((metric) => (
                    <TableRow key={metric}>
                      <TableCell>
                        <Chip
                          label={metric}
                          size="small"
                          color={getMetricColor(metric)}
                        />
                      </TableCell>
                      {selectedVersions.map((versionId) => {
                        const value = comparison.metrics[metric]?.[versionId];
                        return (
                          <TableCell key={versionId} align="right">
                            {typeof value === 'number'
                              ? value.toFixed(4)
                              : 'N/A'}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Grid>

          {/* Line Chart */}
          <Grid item xs={12}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Metrics Comparison
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparison.timeSeriesData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    {selectedVersions.map((versionId, index) => (
                      <Line
                        key={versionId}
                        type="monotone"
                        dataKey={versionId}
                        stroke={`hsl(${(index * 360) / selectedVersions.length}, 70%, 50%)`}
                        activeDot={{ r: 8 }}
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
          </Grid>

          {/* Radar Chart */}
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Performance Radar
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={comparison.radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis />
                    {selectedVersions.map((versionId, index) => (
                      <Radar
                        key={versionId}
                        name={versions.find((v) => v.id === versionId)?.name}
                        dataKey={versionId}
                        stroke={`hsl(${(index * 360) / selectedVersions.length}, 70%, 50%)`}
                        fill={`hsl(${(index * 360) / selectedVersions.length}, 70%, 50%)`}
                        fillOpacity={0.3}
                      />
                    ))}
                    <Legend />
                  </RadarChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
          </Grid>

          {/* Bar Chart */}
          <Grid item xs={12} md={6}>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Metric Distribution
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparison.distributionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    {selectedVersions.map((versionId, index) => (
                      <Bar
                        key={versionId}
                        dataKey={versionId}
                        name={versions.find((v) => v.id === versionId)?.name}
                        fill={`hsl(${(index * 360) / selectedVersions.length}, 70%, 50%)`}
                      />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
} 