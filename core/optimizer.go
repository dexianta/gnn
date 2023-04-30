package core

type MomentumOptim struct {
	lr       float64
	m        float64
	vs       []*V
	velocity []float64
}

func NewMomentumOptim(vs []*V, lr, m float64) MomentumOptim {
	return MomentumOptim{
		lr:       lr,
		vs:       vs,
		m:        m,
		velocity: make([]float64, len(vs)),
	}
}

func (m *MomentumOptim) Step() {
	for i := range m.velocity {
		m.velocity[i] = m.velocity[i]*m.m + m.lr*m.vs[i].grad
		m.vs[i].data -= m.velocity[i]
	}
}

func (m MomentumOptim) ZeroGrad() {
	for _, v := range m.vs {
		v.grad = 0
	}
}
