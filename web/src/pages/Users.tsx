import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Grid,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Tooltip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
  Chip,
  Avatar,
  Switch,
  FormControlLabel,
  Alert,
  Snackbar,
  TablePagination,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Key as KeyIcon,
  Block as BlockIcon,
  CheckCircle as ActiveIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { userApi } from '@/api/client';

interface UserDialogProps {
  open: boolean;
  onClose: () => void;
  user?: User;
}

function UserDialog({ open, onClose, user }: UserDialogProps) {
  const [username, setUsername] = useState(user?.username || '');
  const [email, setEmail] = useState(user?.email || '');
  const [role, setRole] = useState(user?.role || 'user');
  const [password, setPassword] = useState('');
  const queryClient = useQueryClient();

  const { mutate: saveUser, isLoading } = useMutation({
    mutationFn: user
      ? (data: Partial<User>) => userApi.updateUser(user.id, data)
      : (data: Partial<User>) => userApi.createUser({ ...data, password }),
    onSuccess: () => {
      queryClient.invalidateQueries(['users']);
      onClose();
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    saveUser({ username, email, role });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>
          {user ? 'Edit User' : 'Create New User'}
        </DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ mt: 2 }}>
            <TextField
              label="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              fullWidth
              required
            />
            <TextField
              label="Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              fullWidth
              required
            />
            {!user && (
              <TextField
                label="Password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                fullWidth
                required
              />
            )}
            <FormControl fullWidth>
              <InputLabel>Role</InputLabel>
              <Select
                value={role}
                label="Role"
                onChange={(e) => setRole(e.target.value)}
              >
                <MenuItem value="admin">Admin</MenuItem>
                <MenuItem value="manager">Manager</MenuItem>
                <MenuItem value="user">User</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={isLoading || !username || !email || (!user && !password)}
          >
            {isLoading ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}

interface PermissionDialogProps {
  open: boolean;
  onClose: () => void;
  user: User;
}

function PermissionDialog({ open, onClose, user }: PermissionDialogProps) {
  const [permissions, setPermissions] = useState(user.permissions || {});
  const queryClient = useQueryClient();

  const { mutate: updatePermissions, isLoading } = useMutation({
    mutationFn: (data: Record<string, boolean>) =>
      userApi.updatePermissions(user.id, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['users']);
      onClose();
    },
  });

  const handleTogglePermission = (key: string) => {
    setPermissions((prev) => ({
      ...prev,
      [key]: !prev[key],
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    updatePermissions(permissions);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Edit Permissions</DialogTitle>
        <DialogContent>
          <Stack spacing={2} sx={{ mt: 2 }}>
            <Typography variant="subtitle2" gutterBottom>
              User: {user.username}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Configure user permissions for different features and operations.
            </Typography>
            <Card variant="outlined">
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="subtitle2">Models</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.create_models}
                            onChange={() =>
                              handleTogglePermission('create_models')
                            }
                          />
                        }
                        label="Create Models"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.delete_models}
                            onChange={() =>
                              handleTogglePermission('delete_models')
                            }
                          />
                        }
                        label="Delete Models"
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </CardContent>
            </Card>
            <Card variant="outlined">
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="subtitle2">Datasets</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.create_datasets}
                            onChange={() =>
                              handleTogglePermission('create_datasets')
                            }
                          />
                        }
                        label="Create Datasets"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.delete_datasets}
                            onChange={() =>
                              handleTogglePermission('delete_datasets')
                            }
                          />
                        }
                        label="Delete Datasets"
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </CardContent>
            </Card>
            <Card variant="outlined">
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="subtitle2">Training</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.start_training}
                            onChange={() =>
                              handleTogglePermission('start_training')
                            }
                          />
                        }
                        label="Start Training"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.stop_training}
                            onChange={() =>
                              handleTogglePermission('stop_training')
                            }
                          />
                        }
                        label="Stop Training"
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </CardContent>
            </Card>
            <Card variant="outlined">
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="subtitle2">System</Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.view_metrics}
                            onChange={() =>
                              handleTogglePermission('view_metrics')
                            }
                          />
                        }
                        label="View Metrics"
                      />
                    </Grid>
                    <Grid item xs={6}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={permissions.manage_settings}
                            onChange={() =>
                              handleTogglePermission('manage_settings')
                            }
                          />
                        }
                        label="Manage Settings"
                      />
                    </Grid>
                  </Grid>
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={isLoading}
          >
            {isLoading ? 'Saving...' : 'Save'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
}

export default function Users() {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [userDialogOpen, setUserDialogOpen] = useState(false);
  const [permissionDialogOpen, setPermissionDialogOpen] = useState(false);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');
  const queryClient = useQueryClient();

  // Fetch users
  const { data: users } = useQuery({
    queryKey: ['users', page, rowsPerPage],
    queryFn: () => userApi.listUsers({ page, per_page: rowsPerPage }),
  });

  // Delete user mutation
  const { mutate: deleteUser } = useMutation({
    mutationFn: userApi.deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries(['users']);
      setMessage('User deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete user');
      setSeverity('error');
    },
  });

  // Toggle user status mutation
  const { mutate: toggleStatus } = useMutation({
    mutationFn: (userId: string) => userApi.toggleUserStatus(userId),
    onSuccess: () => {
      queryClient.invalidateQueries(['users']);
      setMessage('User status updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update user status');
      setSeverity('error');
    },
  });

  const handleDelete = (userId: string) => {
    if (confirm('Are you sure you want to delete this user?')) {
      deleteUser(userId);
    }
  };

  if (!users) {
    return null;
  }

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
        <Typography variant="h5">Users</Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setSelectedUser(null);
            setUserDialogOpen(true);
          }}
        >
          New User
        </Button>
      </Box>

      {/* Users Table */}
      <Card>
        <CardContent>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>User</TableCell>
                  <TableCell>Role</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created At</TableCell>
                  <TableCell>Last Login</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {users.items.map((user) => (
                  <TableRow key={user.id}>
                    <TableCell>
                      <Stack direction="row" spacing={2} alignItems="center">
                        <Avatar
                          sx={{
                            bgcolor: user.is_active ? 'primary.main' : 'grey.500',
                          }}
                        >
                          {user.username[0].toUpperCase()}
                        </Avatar>
                        <Box>
                          <Typography variant="body1">
                            {user.username}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            {user.email}
                          </Typography>
                        </Box>
                      </Stack>
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={user.role}
                        color={
                          user.role === 'admin'
                            ? 'error'
                            : user.role === 'manager'
                            ? 'warning'
                            : 'default'
                        }
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={user.is_active ? 'Active' : 'Inactive'}
                        color={user.is_active ? 'success' : 'default'}
                        size="small"
                        icon={user.is_active ? <ActiveIcon /> : <BlockIcon />}
                      />
                    </TableCell>
                    <TableCell>
                      {format(new Date(user.created_at), 'yyyy-MM-dd HH:mm:ss')}
                    </TableCell>
                    <TableCell>
                      {user.last_login_at
                        ? format(
                            new Date(user.last_login_at),
                            'yyyy-MM-dd HH:mm:ss'
                          )
                        : '-'}
                    </TableCell>
                    <TableCell align="right">
                      <Stack direction="row" spacing={1} justifyContent="flex-end">
                        <Tooltip title="Edit Permissions">
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedUser(user);
                              setPermissionDialogOpen(true);
                            }}
                          >
                            <KeyIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Edit">
                          <IconButton
                            size="small"
                            onClick={() => {
                              setSelectedUser(user);
                              setUserDialogOpen(true);
                            }}
                          >
                            <EditIcon />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title={user.is_active ? 'Deactivate' : 'Activate'}>
                          <IconButton
                            size="small"
                            color={user.is_active ? 'error' : 'success'}
                            onClick={() => toggleStatus(user.id)}
                          >
                            {user.is_active ? <BlockIcon /> : <ActiveIcon />}
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Delete">
                          <IconButton
                            size="small"
                            color="error"
                            onClick={() => handleDelete(user.id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Tooltip>
                      </Stack>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            component="div"
            count={users.total}
            page={page}
            onPageChange={(_, newPage) => setPage(newPage)}
            rowsPerPage={rowsPerPage}
            onRowsPerPageChange={(e) =>
              setRowsPerPage(parseInt(e.target.value, 10))
            }
          />
        </CardContent>
      </Card>

      {/* Dialogs */}
      <UserDialog
        open={userDialogOpen}
        onClose={() => {
          setUserDialogOpen(false);
          setSelectedUser(null);
        }}
        user={selectedUser || undefined}
      />
      {selectedUser && (
        <PermissionDialog
          open={permissionDialogOpen}
          onClose={() => {
            setPermissionDialogOpen(false);
            setSelectedUser(null);
          }}
          user={selectedUser}
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