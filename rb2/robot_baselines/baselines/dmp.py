import torch
import numpy as np


class DMPIntegrator():
    def __init__(self, rbf='gaussian', only_g=False, az=False):
        a = 1
        self.rbf = rbf
        self.only_g = only_g
        self.az = az
        # self.x = 1


    def forward(self, inputs, parameters, param_gradients, scaling, y0, dy0, goal=None, w=None, vel=False):
        dim = int(parameters[0].item())
        k = dim
        N = int(parameters[1].item())
        division = k*(N + 2)
        inputs_np = inputs
        # w = torch.cat((inputs_np[:, range(2*dim, (2*dim + N*dim),2)], inputs_np[:,range(1+2*dim,(2*dim + N*dim), 2)]), 1).view(-1, N)
        if goal is not None:
            goal = goal
            w = w
        else:
            w = inputs_np[:, dim:dim*(N + 1)]
            goal = inputs_np[:, :dim]

        if self.az:
            alpha_z = inputs[:, -1]
            t = y0.shape[0] // inputs.shape[0]
            alpha_z = alpha_z.repeat(t, 1).transpose(0, 1).reshape(inputs.shape[0], -1)
            alpha_z = alpha_z.contiguous().view(alpha_z.shape[0] * alpha_z.shape[1], )

        w = w.reshape(-1, N)

        if self.only_g:
            w = torch.zeros_like(w)
        if vel:
            dy0 = torch.zeros_like(y0)

        goal = goal.contiguous().view(goal.shape[0]*goal.shape[1], )

        if self.az:
            X, dX, ddX = integrate(parameters, w, y0, dy0, goal, 1, rbf=self.rbf, az=True, alpha_z=alpha_z)
        else:
            X, dX, ddX = integrate(parameters, w, y0, dy0, goal, 1, rbf=self.rbf)
        return inputs.new(X), inputs.new(dX), inputs.new(ddX)


    def forward_not_int(self, inputs, parameters, param_gradients, scaling, y0, dy0, goal=None, w=None, vel=False):
        dim = int(parameters[0].item())
        k = dim
        N = int(parameters[1].item())
        division = k*(N + 2)
        inputs_np = inputs
        if goal is not None:
            goal = goal
            w = w
        else:
            w = inputs_np[:, dim:dim*(N + 1)]
            goal = inputs_np[:, :dim]
        w = w.reshape(-1, N)
        if vel:
            dy0 = torch.zeros_like(y0)
        goal = goal.contiguous().view(goal.shape[0]*goal.shape[1], )
        return parameters, w, y0, dy0, goal, 1

    def first_step(self, w, parameters, scaling, y0, dy0, l, tau=1):
        data = parameters
        y = y0
        self.y0 = y0
        z = dy0 * tau
        self.x = 1
        self.N = int(data[1].item())
        self.dt = data[3].item()
        self.a_x = data[4].item()
        self.a_z = data[5].item()
        self.b_z = self.a_z / 4
        self.h = data[(6+self.N):(6+self.N*2)]
        self.c = data[6:(6+self.N)]
        self.num_steps = int(data[2].item())-1
        self.i = 0
        self.w = w.reshape(-1, self.N)
        self.tau = tau
        self.l = l

    def step(self, g, y, dy):
        g = g.reshape(-1, 1)[:, 0]
        z = dy*self.tau
        dt = self.dt
        for _ in range(self.l):
            dx = (-self.a_x * self.x) / self.tau
            self.x = self.x + dx * dt
            psi = torch.exp(-self.h * torch.pow((self.x - self.c), 2))
            fx = torch.mv(self.w, psi)*self.x*(g - self.y0) / torch.sum(psi)
            dz = self.a_z * (self.b_z * (g - y) - z) + fx
            dy = z
            dz = dz / self.tau
            dy = dy / self.tau
            y = y + dy * dt
            z = z + dz * dt
        self.i += 1
        return y, dy, dz


def integrate(data, w, y0, dy0, goal, tau, rbf='gaussian', az=False, alpha_z=None):
    y = y0
    z = dy0 * tau
    x = 1
    if w.is_cuda:
        Y = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
        dY = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
        ddY = torch.cuda.FloatTensor(w.shape[0], int(data[2].item())).fill_(0)
    else:
        Y = torch.zeros((w.shape[0],int(data[2].item())))
        dY = torch.zeros((w.shape[0],int(data[2].item())))
        ddY = torch.zeros((w.shape[0],int(data[2].item())))
    Y[:, 0] = y
    dY[:, 0] = dy0
    ddY[:, 0] = z
    N = int(data[1].item())
    dt = data[3].item()
    a_x = data[4].item()
    a_z = data[5].item()
    if az:
        a_z = alpha_z
        a_z = torch.clamp(a_z, 0.5, 30)
    b_z = a_z / 4
    h = data[(6+N):(6+N*2)]
    c = data[6:(6+N)]
    for i in range(0, int(data[2].item())-1):
        dx = (-a_x * x) / tau
        x = x + dx * dt
        eps = torch.pow((x - c), 2)
        if rbf == 'gaussian':
            psi = torch.exp(-h * eps)
        if rbf == 'multiquadric':
            psi = torch.sqrt(1 + h * eps)
        if rbf == 'inverse_quadric':
            psi = 1/(1 + h*eps)
        if rbf == 'inverse_multiquadric':
            psi = 1/torch.sqrt(1 + h * eps)
        if rbf == 'linear':
            psi = h * eps
        # psi = torch.exp(-h * torch.pow((x - c), 2))
        fx = torch.mv(w, psi)*x*(goal-y0) / torch.sum(psi)
        dz = a_z * (b_z * (goal - y) - z) + fx
        dy = z
        dz = dz / tau
        dy = dy / tau
        y = y + dy * dt
        z = z + dz * dt
        Y[:, i+1] = y
        dY[:, i+1] = dy
        ddY[:, i+1] = dz
    return Y, dY, ddY


class DMPParameters():
    def __init__(self, N, tau, dt, Dof, scale, a_z=25):
        self.a_z = a_z
        self.a_x = 1
        self.N = N
        c = np.exp(-self.a_x * np.linspace(0, 1, self.N))
        sigma2 = np.ones(self.N) * self.N**1.5 / c / self.a_x
        self.c = torch.from_numpy(c).float()
        self.sigma2 = torch.from_numpy(sigma2).float()
        self.tau = tau
        self.dt = dt
        self.time_steps = int(np.round(self.tau / self.dt))+1
        self.y0 = [0]
        self.dy0 = np.zeros(Dof)
        self.Dof = Dof
        self.Y = torch.zeros((self.time_steps))
        grad = torch.zeros((self.time_steps, 2))
        self.data = {'time_steps':self.time_steps,'c':self.c,'sigma2':self.sigma2,'a_z':self.a_z,'a_x':self.a_x,'dt':self.dt,'Y':self.Y}
        dmp_data = torch.tensor([self.Dof,self.N,self.time_steps,self.dt,self.a_x,self.a_z])
        data_tensor = torch.cat((dmp_data,self.c,self.sigma2),0)
        data_tensor.dy0 = self.dy0
        data_tensor.tau = self.tau
        weights = torch.zeros((1,self.N))
        weights = torch.zeros((1,self.N))
        grad[:, 1], _, _ = integrate(data_tensor, weights, 0, 0, 1, self.tau)
        self.data_tensor = data_tensor
        self.grad_tensor = grad
        self.point_grads = torch.zeros(self.N*2 + 4)
        self.X = np.zeros((self.time_steps, self.Dof))
