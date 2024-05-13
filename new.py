import React from 'react';
import { render, screen } from '@testing-library/react';
import { AppLogo, SlackLink, Menu } from './yourComponentFileName'; // Update with your actual file name

describe('AppLogo Component', () => {
  test('renders with correct alt text', () => {
    render(<AppLogo />);
    const logoAltText = screen.getByAltText('American Express');
    expect(logoAltText).toBeInTheDocument();
  });
});

describe('SlackLink Component', () => {
  test('renders with correct href', () => {
    render(<SlackLink />);
    const slackLink = screen.getByText('Go-Links on Slack');
    expect(slackLink).toHaveAttribute('href', slackUrl);
  });
});

describe('Menu Component', () => {
  test('renders correct number of nav items', () => {
    render(<Menu />);
    const navItems = screen.getAllByRole('listitem');
    expect(navItems).toHaveLength(navRoutesList.length);
  });

  test('renders nav item with active class for current location', () => {
    // Mock window.location
    const mockLocation = { pathname: '/some-route' };
    Object.defineProperty(window, 'location', {
      value: mockLocation,
      writable: true,
    });

    render(<Menu />);
    const activeNavItem = screen.getByText('Your Active Nav Item Text Here');
    expect(activeNavItem).toHaveClass('font-weight-bold');
  });
});
