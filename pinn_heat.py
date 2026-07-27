"""
PINN for 1D Heat Equation – Final Working Version
Solves ∂u/∂t = α ∂²u/∂x² on [0,1]×[0,1] with Dirichlet BCs.
Uses JAX for autodiff and GPU acceleration.
Author: Padraig Hill
"""

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)

# --- Parameters ---
ALPHA = 0.01
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0
N_INTERIOR = 3000
N_BOUNDARY = 200
N_INITIAL = 200
LAYERS = [2, 50, 50, 50, 50, 1]      # Last layer has 1 output
EPOCHS = 5000
LR_INIT = 1e-3

key = jax.random.PRNGKey(42)

# --- Neural Network ---
def init_network(key, layers):
    params = []
    for i in range(len(layers)-1):
        w_key, b_key, key = jax.random.split(key, 3)
        w = jax.random.normal(w_key, (layers[i], layers[i+1])) * jnp.sqrt(2.0 / layers[i])
        b = jax.random.normal(b_key, (layers[i+1],)) * 0.1
        params.append((w, b))
    return params

def forward(params, x):
    """
    Forward pass.
    - x: shape (..., 2) (batched) or (2,) (single point)
    - returns shape (...,)  (scalar for single point, flat for batch)
    """
    for w, b in params[:-1]:
        x = jnp.tanh(jnp.dot(x, w) + b)
    w, b = params[-1]
    out = jnp.dot(x, w) + b          # shape (..., 1)
    return out.squeeze(-1)            # remove last dimension (now (..., ))

# --- Loss functions ---
def pde_loss(params, x, t):
    # Define a scalar function for a single (x,t)
    def u_single(x, t):
        return forward(params, jnp.array([x, t]))  # returns a scalar

    # First derivatives
    u_t = jax.vmap(jax.grad(u_single, argnums=1))(x, t)   # shape (N,)
    u_x = jax.vmap(jax.grad(u_single, argnums=0))(x, t)   # shape (N,)

    # Second derivative: gradient of u_x w.r.t x
    def u_x_single(x, t):
        return jax.grad(u_single, argnums=0)(x, t)        # scalar
    u_xx = jax.vmap(jax.grad(u_x_single, argnums=0))(x, t) # shape (N,)

    residual = u_t - ALPHA * u_xx
    return jnp.mean(residual**2)

def boundary_loss(params, x, t):
    u_pred = forward(params, jnp.stack([x, t], axis=-1))   # shape (N,)
    return jnp.mean(u_pred**2)

def initial_loss(params, x, t):
    u_pred = forward(params, jnp.stack([x, t], axis=-1))   # shape (N,)
    u_exact = jnp.sin(jnp.pi * x)                          # shape (N,)
    return jnp.mean((u_pred - u_exact)**2)

# Weighted loss – tune these weights if needed
PDE_WEIGHT = 1.0
BC_WEIGHT = 1.0
IC_WEIGHT = 10.0

def loss_fn(params, x_int, t_int, x_bc, t_bc, x_ic, t_ic):
    return (PDE_WEIGHT * pde_loss(params, x_int, t_int) +
            BC_WEIGHT * boundary_loss(params, x_bc, t_bc) +
            IC_WEIGHT * initial_loss(params, x_ic, t_ic))

# --- Training ---
def train():
    # Generate data
    key_int, key_bc, key_ic = jax.random.split(key, 3)
    x_int = jax.random.uniform(key_int, (N_INTERIOR,), minval=X_MIN, maxval=X_MAX)
    t_int = jax.random.uniform(key_int, (N_INTERIOR,), minval=T_MIN, maxval=T_MAX)
    t_bc = jax.random.uniform(key_bc, (N_BOUNDARY*2,), minval=T_MIN, maxval=T_MAX)
    x_bc = jnp.concatenate([jnp.zeros(N_BOUNDARY), jnp.ones(N_BOUNDARY)])
    x_ic = jax.random.uniform(key_ic, (N_INITIAL,), minval=X_MIN, maxval=X_MAX)
    t_ic = jnp.zeros(N_INITIAL)

    params = init_network(key, LAYERS)

    # Learning rate schedule
    lr_schedule = optax.cosine_decay_schedule(init_value=LR_INIT, decay_steps=EPOCHS, alpha=1e-5)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss = loss_fn(params, x_int, t_int, x_bc, t_bc, x_ic, t_ic)
        grads = jax.grad(loss_fn, argnums=0)(params, x_int, t_int, x_bc, t_bc, x_ic, t_ic)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses = []
    for epoch in range(EPOCHS):
        params, opt_state, loss = step(params, opt_state)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4e}")
            losses.append(loss)
    
    return params, losses

if __name__ == "__main__":
    params, losses = train()

    # Plot loss
    plt.figure()
    plt.plot(np.arange(0, EPOCHS, 500), losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('loss.png')
    plt.show()

    # Compute and plot solution at t=0.5
    x_test = jnp.linspace(0, 1, 100)
    t_test = jnp.full_like(x_test, 0.5)
    u_pred = forward(params, jnp.stack([x_test, t_test], axis=-1))   # shape (100,)
    u_exact = jnp.exp(-ALPHA * jnp.pi**2 * 0.5) * jnp.sin(jnp.pi * x_test)

    plt.figure()
    plt.plot(x_test, u_pred, label='PINN')
    plt.plot(x_test, u_exact, '--', label='Exact')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u(x, t=0.5)')
    plt.title('Solution at t=0.5')
    plt.savefig('solution.png')
    plt.show()