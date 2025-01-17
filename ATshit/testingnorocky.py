from common_jax_utils.decorators import load_arrays 

class SoundSampler(Sampler):
    """ 
    Sample coordinates from a sound
    """
    sound_fragment: jax.Array #shape (length,)
    fragment_length: int
    window_size: int
    batch_size: int
    
    @load_arrays
    def __init__(self, sound_fragment: jax.Array, fragment_length: int, window_size: int, batch_size: int):
        self.sound_fragment = sound_fragment
        self.fragment_length = fragment_length
        self.window_size = window_size
        self.batch_size = batch_size

    def __call__(self, key : jax.Array) -> tuple[jax.Array, jax.Array]: #output is two (batch_size, window_size) arrays
        #the first one has time points t_0, t_1, ..., t_n
        #the second one has the corresponding pressure values at those time points
        #the time points are sampled uniformly from the interval [0, self.fragment_length-self.window_size]
        #the pressure values are sampled from the sound fragment at the corresponding time points
        # Sample starting points for each window
        start_points = jax.random.uniform(key, shape=(self.batch_size,), minval=0, maxval=self.fragment_length-self.window_size)
        start_points = start_points.astype(jnp.int32)
        
        # Create time points array - each row is a sequence from t to t+window_size
        time_points = jnp.arange(self.window_size)[None,:] + start_points[:,None]
        
        # Get corresponding pressure values from sound fragment
        # vmap over batch dimension to get window for each start point
        pressure_values = jax.vmap(lambda t: self.sound_fragment[t:t+self.window_size])(start_points)
        
        return time_points, pressure_values
    
    
    from common_jax_utils.decorators import load_arrays 

class SoundSampler(Sampler):
    """ 
    Sample coordinates from a sound. Returns batches of time windows and corresponding pressure values
    from a sound fragment for training INRs to represent audio signals.
    """
    sound_fragment: jax.Array #shape (length,)
    fragment_length: int
    window_size: int
    batch_size: int
    
    @load_arrays
    def __init__(self, sound_fragment: jax.Array, fragment_length: int, window_size: int, batch_size: int):
        """
        Initialize the sampler.
        
        Args:
            sound_fragment: Array containing the audio pressure values
            fragment_length: Length of the sound fragment
            window_size: Size of each sampled window
            batch_size: Number of windows to sample in each batch
        """
        self.sound_fragment = sound_fragment
        self.fragment_length = fragment_length
        self.window_size = window_size
        self.batch_size = batch_size

    def __call__(self, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """
        Sample batches of time windows and pressure values.
        
        Args:
            key: PRNG key for random sampling
            
        Returns:
            Tuple of:
            - time_points: Array of shape (batch_size, window_size) containing time indices
            - pressure_values: Array of shape (batch_size, window_size) containing pressure values
        """
        # Sample starting points uniformly from valid range
        start_points = jax.random.uniform(
            key, 
            shape=(self.batch_size,), 
            minval=0, 
            maxval=self.fragment_length - self.window_size
        )
        start_points = jnp.floor(start_points).astype(jnp.int32)
        
        # Create time points array - each row contains window_size sequential points
        time_points = jnp.arange(self.window_size)[None,:] + start_points[:,None]
        time_points = time_points / self.fragment_length  # Normalize to [0,1]
        
        # Get corresponding pressure values using vectorized indexing
        pressure_values = jax.vmap(lambda t: self.sound_fragment[t:t+self.window_size])(start_points)
        
        return time_points, pressure_values
    
    