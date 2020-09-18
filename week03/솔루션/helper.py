import numpy as np
def gradient_descent(grad_func, x_set, y_set, w0,
					learning_rate=0.01, MaxIter=10):
	path = []
	path.append(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# next step
		w1 = w0 - learning_rate * grad
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def newton_method(grad_func, hessian_func, x_set, y_set, w0,
					learning_rate=0.01, MaxIter=10):
	path = []
	path.append(w0)
	for i in range(MaxIter):
		# compute gradient
		grad = grad_func(w0, x_set, y_set)
		# compute hessian
		hessian = hessian_func(x0, x_set, y_set)
		# stopping criteria
		if np.linalg.norm(grad) < 1E-7:
			break
		# set serach direction by Newton method
		dx = np.linalg.solve(hessian, grad)
		# next step
		w1 = w0 - learning_rate * dx
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
	return w0, path

def bfgs_method(grad_func, x_set, y_set, w0,
					learning_rate=0.01, MaxIter=10):
	path = []
	path.append(w0)
	B0 = np.eye(len(w0))
	for i in range(MaxIter):
		grad = grad_func(w0,x_set, y_set)
		if np.linalg.norm(grad) < 1E-7:
			break
		p0 = -np.linalg.solve(B0, grad)
		s0 = learning_rate * p0
		w1 = w0 + s0
		y0 = (grad_func(w1) - grad).reshape(-1,1) # convert to a column vector
		B1 = B0 + np.dot(y0, y0.T) / np.dot(y0.T, s0) \
				- (np.dot(np.dot(B0, s0).reshape(-1,1), np.dot(s0, B0).reshape(-1,1).T)) / np.dot(np.dot(B0, s0), s0)
		# write history of w0
		path.append(w1)
		# update
		w0 = w1
		B0 = B1
	return w0, path

def generate_batches(batch_size, features, labels):
	"""
	Create batches of features and labels
	:param batch_size: The batch size
	:param features: List of features
	:param labels: List of labels
	:return: Batches of (Features, Labels)
	"""
	assert len(features) == len(labels)
	outout_batches = []

	sample_size = len(features)
	for start_i in range(0, sample_size, batch_size):
		end_i = start_i + batch_size
		batch = [features[start_i:end_i], labels[start_i:end_i]]
		outout_batches.append(batch)

	return outout_batches
