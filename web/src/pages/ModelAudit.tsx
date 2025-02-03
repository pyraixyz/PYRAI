import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Paper,
  Stack,
  Button,
  TextField,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  IconButton,
  Chip,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  InputAdornment,
  Alert,
  Snackbar,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  Info as InfoIcon,
  History as HistoryIcon,
  Person as PersonIcon,
  Event as EventIcon,
} from '@mui/icons-material';
import { useQuery, useMutation } from '@tanstack/react-query';
import { format } from 'date-fns';
import { modelApi } from '@/api/client';

interface AuditLog {
  id: string;
  event_type: string;
  user: {
    id: string;
    name: string;
    email: string;
  };
  details: {
    [key: string]: any;
  };
  ip_address: string;
  user_agent: string;
  created_at: string;
}

interface FilterDialogProps {
  open: boolean;
  onClose: () => void;
  onApply: (filters: any) => void;
  currentFilters: any;
}

function FilterDialog({
  open,
  onClose,
  onApply,
  currentFilters,
}: FilterDialogProps) {
  const [filters, setFilters] = useState(currentFilters);

  const handleApply = () => {
    onApply(filters);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Filter Audit Logs</DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <FormControl fullWidth>
            <InputLabel>Event Type</InputLabel>
            <Select
              value={filters.event_type || ''}
              label="Event Type"
              onChange={(e) =>
                setFilters({ ...filters, event_type: e.target.value })
              }
            >
              <MenuItem value="">All</MenuItem>
              <MenuItem value="model.create">Model Created</MenuItem>
              <MenuItem value="model.update">Model Updated</MenuItem>
              <MenuItem value="model.delete">Model Deleted</MenuItem>
              <MenuItem value="version.create">Version Created</MenuItem>
              <MenuItem value="version.deploy">Version Deployed</MenuItem>
              <MenuItem value="permission.update">Permission Updated</MenuItem>
              <MenuItem value="setting.update">Setting Updated</MenuItem>
            </Select>
          </FormControl>

          <TextField
            label="User"
            value={filters.user || ''}
            onChange={(e) => setFilters({ ...filters, user: e.target.value })}
            fullWidth
          />

          <Stack direction="row" spacing={2}>
            <TextField
              label="Start Date"
              type="date"
              value={filters.start_date || ''}
              onChange={(e) =>
                setFilters({ ...filters, start_date: e.target.value })
              }
              InputLabelProps={{ shrink: true }}
              fullWidth
            />
            <TextField
              label="End Date"
              type="date"
              value={filters.end_date || ''}
              onChange={(e) =>
                setFilters({ ...filters, end_date: e.target.value })
              }
              InputLabelProps={{ shrink: true }}
              fullWidth
            />
          </Stack>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleApply} variant="contained">
          Apply Filters
        </Button>
      </DialogActions>
    </Dialog>
  );
}

interface LogDetailsDialogProps {
  open: boolean;
  onClose: () => void;
  log: AuditLog | null;
}

function LogDetailsDialog({ open, onClose, log }: LogDetailsDialogProps) {
  if (!log) return null;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Audit Log Details</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2">Event Information</Typography>
              <Stack direction="row" spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  Event Type:
                </Typography>
                <Typography variant="body2">{log.event_type}</Typography>
              </Stack>
              <Stack direction="row" spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  Timestamp:
                </Typography>
                <Typography variant="body2">
                  {format(new Date(log.created_at), 'yyyy-MM-dd HH:mm:ss')}
                </Typography>
              </Stack>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2">User Information</Typography>
              <Stack direction="row" spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  Name:
                </Typography>
                <Typography variant="body2">{log.user.name}</Typography>
              </Stack>
              <Stack direction="row" spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  Email:
                </Typography>
                <Typography variant="body2">{log.user.email}</Typography>
              </Stack>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2">Request Details</Typography>
              <Stack direction="row" spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  IP Address:
                </Typography>
                <Typography variant="body2">{log.ip_address}</Typography>
              </Stack>
              <Stack direction="row" spacing={2}>
                <Typography variant="body2" color="text.secondary">
                  User Agent:
                </Typography>
                <Typography variant="body2">{log.user_agent}</Typography>
              </Stack>
            </Stack>
          </Paper>

          <Paper variant="outlined" sx={{ p: 2 }}>
            <Stack spacing={1}>
              <Typography variant="subtitle2">Event Details</Typography>
              <Box
                component="pre"
                sx={{
                  p: 2,
                  bgcolor: 'background.default',
                  borderRadius: 1,
                  overflow: 'auto',
                }}
              >
                {JSON.stringify(log.details, null, 2)}
              </Box>
            </Stack>
          </Paper>
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
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchQuery, setSearchQuery] = useState('');
  const [filters, setFilters] = useState({});
  const [filterDialogOpen, setFilterDialogOpen] = useState(false);
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
  const { data: auditLogs, refetch } = useQuery({
    queryKey: ['model', modelId, 'audit', { page, rowsPerPage, searchQuery, filters }],
    queryFn: () =>
      modelApi.getAuditLogs(modelId!, {
        page,
        per_page: rowsPerPage,
        search: searchQuery,
        ...filters,
      }),
    enabled: !!modelId,
  });

  // Export logs mutation
  const { mutate: exportLogs } = useMutation({
    mutationFn: () => modelApi.exportAuditLogs(modelId!),
    onSuccess: () => {
      setMessage('Audit logs exported successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to export audit logs');
      setSeverity('error');
    },
  });

  if (!model || !auditLogs) {
    return null;
  }

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleSearch = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(event.target.value);
    setPage(0);
  };

  const handleApplyFilters = (newFilters: any) => {
    setFilters(newFilters);
    setPage(0);
  };

  const getEventTypeColor = (eventType: string) => {
    switch (eventType) {
      case 'model.create':
        return 'success';
      case 'model.update':
        return 'info';
      case 'model.delete':
        return 'error';
      case 'version.create':
        return 'primary';
      case 'version.deploy':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Audit Logs
        </Typography>
        <Typography variant="body2" color="text.secondary">
          View operation history for {model.name}
        </Typography>
      </Box>

      {/* Actions */}
      <Stack
        direction="row"
        spacing={2}
        alignItems="center"
        justifyContent="space-between"
        sx={{ mb: 3 }}
      >
        <TextField
          placeholder="Search logs..."
          value={searchQuery}
          onChange={handleSearch}
          sx={{ width: 300 }}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <SearchIcon />
              </InputAdornment>
            ),
          }}
        />
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            startIcon={<FilterIcon />}
            onClick={() => setFilterDialogOpen(true)}
          >
            Filter
          </Button>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={() => exportLogs()}
          >
            Export
          </Button>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
          >
            Refresh
          </Button>
        </Stack>
      </Stack>

      {/* Logs Table */}
      <Card>
        <CardContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Event Type</TableCell>
                  <TableCell>User</TableCell>
                  <TableCell>Timestamp</TableCell>
                  <TableCell>IP Address</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {auditLogs.items.map((log) => (
                  <TableRow key={log.id}>
                    <TableCell>
                      <Chip
                        label={log.event_type}
                        size="small"
                        color={getEventTypeColor(log.event_type)}
                      />
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <PersonIcon fontSize="small" />
                        <div>
                          <Typography variant="body2">{log.user.name}</Typography>
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            display="block"
                          >
                            {log.user.email}
                          </Typography>
                        </div>
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Stack direction="row" spacing={1} alignItems="center">
                        <EventIcon fontSize="small" />
                        <Typography variant="body2">
                          {format(
                            new Date(log.created_at),
                            'yyyy-MM-dd HH:mm:ss'
                          )}
                        </Typography>
                      </Stack>
                    </TableCell>
                    <TableCell>{log.ip_address}</TableCell>
                    <TableCell align="right">
                      <Tooltip title="View Details">
                        <IconButton
                          size="small"
                          onClick={() => setSelectedLog(log)}
                        >
                          <InfoIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            component="div"
            count={auditLogs.total}
            page={page}
            onPageChange={handleChangePage}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={handleChangeRowsPerPage}
          />
        </CardContent>
      </Card>

      {/* Filter Dialog */}
      <FilterDialog
        open={filterDialogOpen}
        onClose={() => setFilterDialogOpen(false)}
        onApply={handleApplyFilters}
        currentFilters={filters}
      />

      {/* Log Details Dialog */}
      <LogDetailsDialog
        open={!!selectedLog}
        onClose={() => setSelectedLog(null)}
        log={selectedLog}
      />

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