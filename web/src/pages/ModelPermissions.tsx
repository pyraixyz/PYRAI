import { useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Typography,
  Grid,
  Stack,
  Button,
  Paper,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Checkbox,
  FormControlLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Snackbar,
  Autocomplete,
  Avatar,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  Switch,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon,
  Security as SecurityIcon,
  Person as PersonIcon,
  Group as GroupIcon,
  Business as OrganizationIcon,
  Public as PublicIcon,
  Lock as LockIcon,
  History as HistoryIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { format } from 'date-fns';
import { modelApi } from '@/api/client';

interface Permission {
  id: string;
  user_id?: string;
  group_id?: string;
  organization_id?: string;
  role: 'owner' | 'admin' | 'editor' | 'viewer';
  created_at: string;
  updated_at: string;
  created_by: {
    id: string;
    name: string;
  };
}

interface User {
  id: string;
  name: string;
  email: string;
  avatar_url?: string;
}

interface Group {
  id: string;
  name: string;
  description: string;
  member_count: number;
}

interface Organization {
  id: string;
  name: string;
  domain: string;
  logo_url?: string;
}

interface PermissionDialogProps {
  open: boolean;
  onClose: () => void;
  onSave: (data: Partial<Permission>) => void;
  permission?: Permission;
  users: User[];
  groups: Group[];
  organizations: Organization[];
}

function PermissionDialog({
  open,
  onClose,
  onSave,
  permission,
  users,
  groups,
  organizations,
}: PermissionDialogProps) {
  const [formData, setFormData] = useState(
    permission || {
      role: 'viewer',
    }
  );

  const handleSave = () => {
    onSave(formData);
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        {permission ? 'Edit Permission' : 'Add Permission'}
      </DialogTitle>
      <DialogContent>
        <Stack spacing={3} sx={{ mt: 1 }}>
          <FormControl fullWidth>
            <InputLabel>Type</InputLabel>
            <Select
              value={
                formData.user_id
                  ? 'user'
                  : formData.group_id
                  ? 'group'
                  : 'organization'
              }
              label="Type"
              onChange={(e) => {
                setFormData({
                  ...formData,
                  user_id: undefined,
                  group_id: undefined,
                  organization_id: undefined,
                });
              }}
            >
              <MenuItem value="user">User</MenuItem>
              <MenuItem value="group">Group</MenuItem>
              <MenuItem value="organization">Organization</MenuItem>
            </Select>
          </FormControl>

          {formData.user_id === undefined &&
            formData.group_id === undefined &&
            formData.organization_id === undefined && (
              <Autocomplete
                options={
                  formData.user_id !== undefined
                    ? users
                    : formData.group_id !== undefined
                    ? groups
                    : organizations
                }
                getOptionLabel={(option) => option.name}
                renderOption={(props, option) => (
                  <Box
                    component="li"
                    sx={{ '& > img': { mr: 2, flexShrink: 0 } }}
                    {...props}
                  >
                    {formData.user_id !== undefined ? (
                      <Avatar
                        src={option.avatar_url}
                        sx={{ width: 24, height: 24, mr: 1 }}
                      >
                        {option.name[0]}
                      </Avatar>
                    ) : formData.group_id !== undefined ? (
                      <GroupIcon sx={{ mr: 1 }} />
                    ) : (
                      <OrganizationIcon sx={{ mr: 1 }} />
                    )}
                    {option.name}
                  </Box>
                )}
                renderInput={(params) => (
                  <TextField
                    {...params}
                    label={
                      formData.user_id !== undefined
                        ? 'User'
                        : formData.group_id !== undefined
                        ? 'Group'
                        : 'Organization'
                    }
                  />
                )}
                onChange={(_, value) => {
                  if (value) {
                    setFormData({
                      ...formData,
                      user_id:
                        formData.user_id !== undefined ? value.id : undefined,
                      group_id:
                        formData.group_id !== undefined ? value.id : undefined,
                      organization_id:
                        formData.organization_id !== undefined
                          ? value.id
                          : undefined,
                    });
                  }
                }}
              />
            )}

          <FormControl fullWidth>
            <InputLabel>Role</InputLabel>
            <Select
              value={formData.role}
              label="Role"
              onChange={(e) =>
                setFormData({
                  ...formData,
                  role: e.target.value as Permission['role'],
                })
              }
            >
              <MenuItem value="owner">Owner</MenuItem>
              <MenuItem value="admin">Admin</MenuItem>
              <MenuItem value="editor">Editor</MenuItem>
              <MenuItem value="viewer">Viewer</MenuItem>
            </Select>
          </FormControl>
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={
            !formData.role ||
            (!formData.user_id &&
              !formData.group_id &&
              !formData.organization_id)
          }
        >
          {permission ? 'Update' : 'Add'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default function ModelPermissions() {
  const { modelId } = useParams<{ modelId: string }>();
  const queryClient = useQueryClient();
  const [permissionDialogOpen, setPermissionDialogOpen] = useState(false);
  const [selectedPermission, setSelectedPermission] = useState<Permission>();
  const [message, setMessage] = useState<string>('');
  const [severity, setSeverity] = useState<'success' | 'error'>('success');

  // Fetch model
  const { data: model } = useQuery({
    queryKey: ['model', modelId],
    queryFn: () => modelApi.getModel(modelId!),
    enabled: !!modelId,
  });

  // Fetch permissions
  const { data: permissions } = useQuery({
    queryKey: ['model', modelId, 'permissions'],
    queryFn: () => modelApi.getModelPermissions(modelId!),
    enabled: !!modelId,
  });

  // Fetch users
  const { data: users } = useQuery({
    queryKey: ['users'],
    queryFn: () => modelApi.getUsers(),
  });

  // Fetch groups
  const { data: groups } = useQuery({
    queryKey: ['groups'],
    queryFn: () => modelApi.getGroups(),
  });

  // Fetch organizations
  const { data: organizations } = useQuery({
    queryKey: ['organizations'],
    queryFn: () => modelApi.getOrganizations(),
  });

  // Create permission mutation
  const { mutate: createPermission } = useMutation({
    mutationFn: (data: Partial<Permission>) =>
      modelApi.createModelPermission(modelId!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'permissions']);
      setMessage('Permission added successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to add permission');
      setSeverity('error');
    },
  });

  // Update permission mutation
  const { mutate: updatePermission } = useMutation({
    mutationFn: (data: Partial<Permission>) =>
      modelApi.updateModelPermission(modelId!, data.id!, data),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'permissions']);
      setMessage('Permission updated successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to update permission');
      setSeverity('error');
    },
  });

  // Delete permission mutation
  const { mutate: deletePermission } = useMutation({
    mutationFn: (permissionId: string) =>
      modelApi.deleteModelPermission(modelId!, permissionId),
    onSuccess: () => {
      queryClient.invalidateQueries(['model', modelId, 'permissions']);
      setMessage('Permission deleted successfully');
      setSeverity('success');
    },
    onError: () => {
      setMessage('Failed to delete permission');
      setSeverity('error');
    },
  });

  if (!model || !permissions || !users || !groups || !organizations) {
    return null;
  }

  const handlePermissionSave = (data: Partial<Permission>) => {
    if (selectedPermission) {
      updatePermission({ ...data, id: selectedPermission.id });
    } else {
      createPermission(data);
    }
    setPermissionDialogOpen(false);
    setSelectedPermission(undefined);
  };

  const handlePermissionDelete = (permissionId: string) => {
    if (
      confirm(
        'Are you sure you want to delete this permission? This action cannot be undone.'
      )
    ) {
      deletePermission(permissionId);
    }
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'owner':
        return 'error';
      case 'admin':
        return 'warning';
      case 'editor':
        return 'info';
      case 'viewer':
        return 'success';
      default:
        return 'default';
    }
  };

  const getEntityIcon = (permission: Permission) => {
    if (permission.user_id) {
      const user = users.find((u) => u.id === permission.user_id);
      return user?.avatar_url ? (
        <Avatar src={user.avatar_url} sx={{ width: 24, height: 24 }} />
      ) : (
        <PersonIcon />
      );
    } else if (permission.group_id) {
      return <GroupIcon />;
    } else {
      return <OrganizationIcon />;
    }
  };

  const getEntityName = (permission: Permission) => {
    if (permission.user_id) {
      return users.find((u) => u.id === permission.user_id)?.name;
    } else if (permission.group_id) {
      return groups.find((g) => g.id === permission.group_id)?.name;
    } else {
      return organizations.find(
        (o) => o.id === permission.organization_id
      )?.name;
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box sx={{ mb: 3 }}>
        <Typography variant="h5" gutterBottom>
          Permissions
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Manage access permissions for {model.name}
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
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => {
            setSelectedPermission(undefined);
            setPermissionDialogOpen(true);
          }}
        >
          Add Permission
        </Button>
        <Tooltip title="Refresh">
          <IconButton
            onClick={() =>
              queryClient.invalidateQueries(['model', modelId, 'permissions'])
            }
          >
            <RefreshIcon />
          </IconButton>
        </Tooltip>
      </Stack>

      {/* Permissions List */}
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Entity</TableCell>
              <TableCell>Type</TableCell>
              <TableCell>Role</TableCell>
              <TableCell>Added By</TableCell>
              <TableCell>Added At</TableCell>
              <TableCell align="right">Actions</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {permissions.map((permission) => (
              <TableRow key={permission.id}>
                <TableCell>
                  <Stack direction="row" spacing={1} alignItems="center">
                    {getEntityIcon(permission)}
                    <Typography>{getEntityName(permission)}</Typography>
                  </Stack>
                </TableCell>
                <TableCell>
                  <Chip
                    label={
                      permission.user_id
                        ? 'User'
                        : permission.group_id
                        ? 'Group'
                        : 'Organization'
                    }
                    size="small"
                  />
                </TableCell>
                <TableCell>
                  <Chip
                    label={permission.role}
                    color={getRoleColor(permission.role)}
                    size="small"
                  />
                </TableCell>
                <TableCell>{permission.created_by.name}</TableCell>
                <TableCell>
                  {format(new Date(permission.created_at), 'PPp')}
                </TableCell>
                <TableCell align="right">
                  <Stack
                    direction="row"
                    spacing={1}
                    justifyContent="flex-end"
                  >
                    <Tooltip title="Edit">
                      <IconButton
                        onClick={() => {
                          setSelectedPermission(permission);
                          setPermissionDialogOpen(true);
                        }}
                      >
                        <EditIcon />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Delete">
                      <IconButton
                        color="error"
                        onClick={() => handlePermissionDelete(permission.id)}
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

      {/* Permission Dialog */}
      <PermissionDialog
        open={permissionDialogOpen}
        onClose={() => {
          setPermissionDialogOpen(false);
          setSelectedPermission(undefined);
        }}
        onSave={handlePermissionSave}
        permission={selectedPermission}
        users={users}
        groups={groups}
        organizations={organizations}
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