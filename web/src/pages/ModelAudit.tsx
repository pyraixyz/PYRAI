import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Stack,
  Button,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  Snackbar,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
} from '@mui/material';
import {
  Refresh as RefreshIcon,
  Download as DownloadIcon,
  DateRange as DateRangeIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Info as InfoIcon,
  History as HistoryIcon,
  Person as PersonIcon,
  Event as EventIcon,
  Category as CategoryIcon,
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { format, subDays } from 'date-fns';
import { modelApi } from '@/api/client';

interface AuditLog {
  id: string;
  timestamp: string;
  user: {
    id: string;
    name: string;
    email: string;
  };
  action: string;
  category: string;
  details: {
    [key: string]: any;
  };
  status: 'success' | 'failure';
  error_message?: string;
}

interface AuditLogDetailsProps {
  open: boolean;
  onClose: () => void;
  log: AuditLog;
}

const TIME_RANGES = [
  { label: 'Last 24 Hours', value: 1 },
  { label: 'Last 7 Days', value: 7 },
  { label: 'Last 30 Days', value: 30 },
  { label: 'Last 90 Days', value: 90 },
];

const ACTION_CATEGORIES = [
  { label: 'All Categories', value: 'all' },
  { label: 'Model Updates', value: 'model_update' },
  { label: 'Version Control', value: 'version_control' },
  { label: 'Deployments', value: 'deployment' },
  { label: 'API Access', value: 'api_access' },
  { label: 'Security', value: 'security' },
  { label: 'Settings', value: 'settings' },
];

function AuditLogDetails({ open, onClose, log }: AuditLogDetailsProps) {
  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Audit Log Details</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          {/* Basic Info */}
          <Card variant="outlined">
            <CardContent>
              <Stack spacing={2}>
                <Stack direction="row" spacing={2} alignItems="center">
                  <EventIcon color="action" />
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      Timestamp
                    </Typography>
                    <Typography>
                      {format(new Date(log.timestamp), 'PPpp')}
                    </Typography>
                  </Box>
                </Stack>

                <Stack direction="row" spacing={2} alignItems="center">
                  <PersonIcon color="action" />
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      User
                    </Typography>
                    <Typography>{log.user.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {log.user.email}
                    </Typography>
                  </Box>
                </Stack>

                <Stack direction="row" spacing={2} alignItems="center">
                  <CategoryIcon color="action" />
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      Category & Action
                    </Typography>
                    <Stack direction="row" spacing={1} alignItems="center">
                      <Chip
                        label={log.category}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                      <Typography>{log.action}</Typography>
                    </Stack>
                  </Box>
                </Stack>

                <Stack direction="row" spacing={2} alignItems="center">
                  <InfoIcon color="action" />
                  <Box>
                    <Typography variant="subtitle2" color="text.secondary">
                      Status
                    </Typography>
                    <Chip
                      label={log.status}
                      size="small"
                      color={log.status === 'success' ? 'success' : 'error'}
                    />
                    {log.error_message && (
                      <Typography color="error" sx={{ mt: 1 }}>
                        {log.error_message}
                      </Typography>
                    )}
                  </Box>
                </Stack>
              </Stack>
            </CardContent>
          </Card>

          {/* Details */}
          <Card variant="outlined">
            <CardContent>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Operation Details
              </Typography>
              <Paper
                variant="outlined"
                sx={{
                  p: 2,
                  backgroundColor: 'grey.50',
                  fontFamily: 'monospace',
                }}
              >
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                  {JSON.stringify(log.details, null, 2)}
                </pre>
              </Paper>
            </CardContent>
          </Card>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelAudit() {
  const { modelId } = useParams<{ modelId: string }>();
  const [timeRange, setTimeRange] = useState(7);
  const [category, setCategory] = useState('all');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedLog, setSelectedLog] = useState<AuditLog | null>(null);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch audit logs
  const { data: auditLogs, isLoading } = useQuery({
    queryKey: ['model', modelId, 'audit', timeRange, category, search, page, rowsPerPage],
    queryFn: () =>
      modelApi.getModelAuditLogs(modelId!, {
        start_date: format(subDays(new Date(), timeRange), 'yyyy-MM-dd'),
        end_date: format(new Date(), 'yyyy-MM-dd'),
        category: category === 'all' ? undefined : category,
        search: search || undefined,
        page: page + 1,
        per_page: rowsPerPage,
      }),
    enabled: !!modelId,
  });

  if (!model || (!auditLogs && !isLoading)) {
    return null;
  }

  const handleExport = async () => {
    try {
      await modelApi.exportModelAuditLogs(modelId!, {
        start_date: format(subDays(new Date(), timeRange), 'yyyy-MM-dd'),
        end_date: format(new Date(), 'yyyy-MM-dd'),
        category: category === 'all' ? undefined : category,
        search: search || undefined,
      });
      setMessage('Audit logs exported successfully');
      setSeverity('success');
    } catch (error) {
      setMessage('Failed to export audit logs');
      setSeverity('error');
    }
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Audit Logs
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Track and audit all operations performed on {model.name}
        </Typography>
      </Box>

      {/* Controls */}
      <Stack spacing={3} sx={{ mb: 3 }}>
        <Stack
          direction="row"
          spacing={2}
          alignItems="center"
          justifyContent="space-between"
        >
          <Stack direction="row" spacing={2}>
            <FormControl size="small" sx={{ width: 200 }}>
              <InputLabel>Time Range</InputLabel>
              <Select
                value={timeRange}
                label="Time Range"
                onChange={(e) => setTimeRange(e.target.value as number)}
                startAdornment={<DateRangeIcon sx={{ mr: 1 }} />}
              >
                {TIME_RANGES.map((range) => (
                  <MenuItem key={range.value} value={range.value}>
                    {range.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ width: 200 }}>
              <InputLabel>Category</InputLabel>
              <Select
                value={category}
                label="Category"
                onChange={(e) => setCategory(e.target.value)}
                startAdornment={<FilterIcon sx={{ mr: 1 }} />}
              >
                {ACTION_CATEGORIES.map((cat) => (
                  <MenuItem key={cat.value} value={cat.value}>
                    {cat.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <TextField
              size="small"
              placeholder="Search logs..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              InputProps={{
                startAdornment: <SearchIcon sx={{ mr: 1, color: 'action.active' }} />,
              }}
            />
          </Stack>

          <Stack direction="row" spacing={1}>
            <Button
              variant="outlined"
              startIcon={<DownloadIcon />}
              onClick={handleExport}
            >
              Export
            </Button>
            <Tooltip title="Refresh">
              <IconButton>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Stack>
        </Stack>

        {/* Audit Logs Table */}
        <TableContainer component={Paper} variant="outlined">
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>User</TableCell>
                <TableCell>Category</TableCell>
                <TableCell>Action</TableCell>
                <TableCell>Status</TableCell>
                <TableCell align="right">Details</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {isLoading ? (
                <TableRow>
                  <TableCell colSpan={6} sx={{ border: 0 }}>
                    <LinearProgress />
                  </TableCell>
                </TableRow>
              ) : (
                auditLogs?.items.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell>
                      {format(new Date(log.timestamp), 'PPp')}
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <PersonIcon fontSize="small" color="action" />
                        <Box>
                          <Typography variant="body2">{log.user.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {log.user.email}
                          </Typography>
                        </Box>
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={log.category}
                        size="small"
                        color="primary"
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell>{log.action}</TableCell>
                    <TableCell>
                      <Chip
                        label={log.status}
                        size="small"
                        color={log.status === 'success' ? 'success' : 'error'}
                      />
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => setSelectedLog(log)}
                        >
                          <InfoIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
          <TablePagination
            component="div"
            count={auditLogs?.total || 0}
            page={page}
            onPageChange={handleChangePage}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </TableContainer>
      </Stack>

      {/* Audit Log Details Dialog */}
      {selectedLog && (
        <AuditLogDetails
          open={!!selectedLog}
          onClose={() => setSelectedLog(null)}
          log={selectedLog}
        />
      )}

      {/* Message Snackbar */}
      <Snackbar
        open={!!message}
        autoHideDuration={3000}
        onClose={() => setMessage('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert
          onClose={() => setMessage('')}
          severity={severity}
          variant="filled"
          sx={{ width: '100%' }}
        >
          {message}
        </Alert>
      </Snackbar>
    </Box>
  );
} 