import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Card,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import {
  Add as AddIcon,
  Stop as StopIcon,
  Visibility as ViewIcon,
  Delete as DeleteIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { trainingApi } from '@/api/client';
import JobStatusChip from '@/components/JobStatusChip';
import JobProgress from '@/components/JobProgress';
import NewJobDialog from '@/components/NewJobDialog';

export default function Jobs() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [dialogOpen, setDialogOpen] = useState(false);

  // Fetch jobs
  const { data: jobList } = useQuery({
    queryKey: ['jobs', statusFilter],
    queryFn: () =>
      trainingApi.listJobs(statusFilter === 'all' ? undefined : statusFilter),
    refetchInterval: 5000,
  });

  // Stop job mutation
  const { mutate: stopJob } = useMutation({
    mutationFn: trainingApi.stopTraining,
    onSuccess: () => {
      queryClient.invalidateQueries(['jobs']);
    },
  });

  // Cleanup jobs mutation
  const { mutate: cleanupJobs } = useMutation({
    mutationFn: trainingApi.cleanupJobs,
    onSuccess: () => {
      queryClient.invalidateQueries(['jobs']);
    },
  });

  const handleStatusChange = (event: SelectChangeEvent) => {
    setStatusFilter(event.target.value);
  };

  const handleStopJob = (jobId: string) => {
    if (confirm('Are you sure you want to stop this job?')) {
      stopJob(jobId);
    }
  };

  const handleCleanup = () => {
    if (confirm('Are you sure you want to clean up old jobs?')) {
      cleanupJobs();
    }
  };

  return (
    <Box>
      {/* Actions */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 3,
        }}
      >
        <FormControl sx={{ minWidth: 120 }}>
          <InputLabel>Status</InputLabel>
          <Select
            value={statusFilter}
            label="Status"
            onChange={handleStatusChange}
          >
            <MenuItem value="all">All</MenuItem>
            <MenuItem value="pending">Pending</MenuItem>
            <MenuItem value="running">Running</MenuItem>
            <MenuItem value="completed">Completed</MenuItem>
            <MenuItem value="failed">Failed</MenuItem>
            <MenuItem value="interrupted">Interrupted</MenuItem>
          </Select>
        </FormControl>
        <Box>
          <Button
            variant="outlined"
            onClick={handleCleanup}
            sx={{ mr: 1 }}
          >
            Clean Up
          </Button>
          <Button
            variant="contained"
            startIcon={<AddIcon />}
            onClick={() => setDialogOpen(true)}
          >
            New Job
          </Button>
        </Box>
      </Box>

      {/* Jobs Table */}
      <Card>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>ID</TableCell>
                <TableCell>Model</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Progress</TableCell>
                <TableCell>Start Time</TableCell>
                <TableCell>End Time</TableCell>
                <TableCell align="right">Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {jobList?.jobs.map((job) => (
                <TableRow key={job.id}>
                  <TableCell>{job.id}</TableCell>
                  <TableCell>{job.model_id}</TableCell>
                  <TableCell>
                    <JobStatusChip status={job.status} />
                  </TableCell>
                  <TableCell>
                    <JobProgress progress={job.progress} />
                  </TableCell>
                  <TableCell>
                    {format(new Date(job.start_time), 'yyyy-MM-dd HH:mm:ss')}
                  </TableCell>
                  <TableCell>
                    {job.end_time
                      ? format(new Date(job.end_time), 'yyyy-MM-dd HH:mm:ss')
                      : '-'}
                  </TableCell>
                  <TableCell align="right">
                    <Tooltip title="View Details">
                      <IconButton
                        onClick={() => navigate(`/jobs/${job.id}`)}
                        size="small"
                      >
                        <ViewIcon />
                      </IconButton>
                    </Tooltip>
                    {job.status === 'running' && (
                      <Tooltip title="Stop Job">
                        <IconButton
                          onClick={() => handleStopJob(job.id)}
                          size="small"
                          color="error"
                        >
                          <StopIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    {['completed', 'failed', 'interrupted'].includes(
                      job.status
                    ) && (
                      <Tooltip title="Delete Job">
                        <IconButton
                          onClick={() => handleCleanup()}
                          size="small"
                          color="error"
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

      {/* New Job Dialog */}
      <NewJobDialog open={dialogOpen} onClose={() => setDialogOpen(false)} />
    </Box>
  );
} 