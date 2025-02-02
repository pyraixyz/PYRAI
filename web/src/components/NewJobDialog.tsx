import { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
} from '@mui/material';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { trainingApi, TrainingConfig } from '@/api/client';

interface NewJobDialogProps {
  open: boolean;
  onClose: () => void;
}

const defaultConfig: TrainingConfig = {
  model_id: '',
  cpu_cores: 1,
  memory_mb: 1024,
  gpu_devices: 0,
  batch_size: 32,
  learning_rate: 0.001,
  max_epochs: 100,
};

export default function NewJobDialog({ open, onClose }: NewJobDialogProps) {
  const [config, setConfig] = useState<TrainingConfig>(defaultConfig);
  const queryClient = useQueryClient();

  const { mutate: startTraining, isLoading } = useMutation({
    mutationFn: trainingApi.startTraining,
    onSuccess: () => {
      queryClient.invalidateQueries(['jobs']);
      onClose();
      setConfig(defaultConfig);
    },
  });

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement> | SelectChangeEvent
  ) => {
    const { name, value } = e.target;
    setConfig((prev) => ({
      ...prev,
      [name]: name === 'model_id' ? value : Number(value),
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    startTraining(config);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Start New Training Job</DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                name="model_id"
                label="Model ID"
                value={config.model_id}
                onChange={handleChange}
                fullWidth
                required
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>CPU Cores</InputLabel>
                <Select
                  name="cpu_cores"
                  value={config.cpu_cores.toString()}
                  onChange={handleChange}
                  label="CPU Cores"
                >
                  {[1, 2, 4, 8].map((cores) => (
                    <MenuItem key={cores} value={cores}>
                      {cores} {cores === 1 ? 'Core' : 'Cores'}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Memory (MB)</InputLabel>
                <Select
                  name="memory_mb"
                  value={config.memory_mb.toString()}
                  onChange={handleChange}
                  label="Memory (MB)"
                >
                  {[1024, 2048, 4096, 8192].map((memory) => (
                    <MenuItem key={memory} value={memory}>
                      {memory} MB
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>GPU Devices</InputLabel>
                <Select
                  name="gpu_devices"
                  value={config.gpu_devices.toString()}
                  onChange={handleChange}
                  label="GPU Devices"
                >
                  {[0, 1, 2, 4].map((gpus) => (
                    <MenuItem key={gpus} value={gpus}>
                      {gpus} {gpus === 1 ? 'GPU' : 'GPUs'}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth>
                <InputLabel>Batch Size</InputLabel>
                <Select
                  name="batch_size"
                  value={config.batch_size.toString()}
                  onChange={handleChange}
                  label="Batch Size"
                >
                  {[16, 32, 64, 128].map((size) => (
                    <MenuItem key={size} value={size}>
                      {size}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                name="learning_rate"
                label="Learning Rate"
                type="number"
                value={config.learning_rate}
                onChange={handleChange}
                inputProps={{ step: 0.0001, min: 0 }}
                fullWidth
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <TextField
                name="max_epochs"
                label="Max Epochs"
                type="number"
                value={config.max_epochs}
                onChange={handleChange}
                inputProps={{ step: 1, min: 1 }}
                fullWidth
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={isLoading || !config.model_id}
          >
            {isLoading ? 'Starting...' : 'Start Training'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
} 